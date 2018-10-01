import argparse
import json
import logging
import os

import numpy as np

from datasets import YelpData, SyntheticData, synthetic_config
from utils import fit_bernoulli, fit_simple, impute, fit_logis, logis_proba, gformula


def textless_mi(truth, mask, k):
  c, a, y = truth

  obs_i = [i for i in range(len(mask)) if not mask[i]]
  missing_i = [i for i in range(len(mask)) if mask[i]]
  a_imputed = impute(
      (c[obs_i], y[obs_i]),
      a[obs_i],
      (c[missing_i], y[missing_i]))

  def get_imputed_values(imputed_probs):
    vals = []
    for i in range(len(missing_i)):
      w = np.random.choice([0, 1], 1, p=a_imputed[i])
      vals.append(w)
    return np.squeeze(np.array(vals))

  resamples = []
  for i in range(k):
    imp_a = a.copy()
    imp_a[missing_i] = get_imputed_values(a_imputed)

    imp_pyac = fit_simple((c, imp_a), y)
    resamples.append(imp_pyac)

  pyac = {key: 0 for key in resamples[0]}
  for key in pyac:
    for resample in resamples:
      pyac[key] += resample[key]
    pyac[key] = pyac[key] / k

  return pyac


def bad_mis(truth, t, mask, k):
  c, a, y = truth

  obs_i = [i for i in range(len(mask)) if not mask[i]]
  missing_i = [i for i in range(len(mask)) if mask[i]]
  a_imputed = impute(
      (c[obs_i], t[obs_i]),
      a[obs_i],
      (c[missing_i], t[missing_i]))

  def get_imputed_values(imputed_probs):
    vals = []
    for i in range(len(missing_i)):
      w = np.random.choice([0, 1], 1, p=a_imputed[i])
      vals.append(w)
    return np.squeeze(np.array(vals))

  resamples = []
  for i in range(k):
    imp_a = a.copy()
    imp_a[missing_i] = get_imputed_values(a_imputed)

    imp_pyac = fit_simple((c, imp_a), y)
    resamples.append(imp_pyac)

  pyac = {key: 0 for key in resamples[0]}
  for key in pyac:
    for resample in resamples:
      pyac[key] += resample[key]
    pyac[key] = pyac[key] / k

  return pyac


def mi(truth, t, mask, k):
  c, a, y = truth

  obs_i = [i for i in range(len(mask)) if not mask[i]]
  missing_i = [i for i in range(len(mask)) if mask[i]]

  a_imputed = impute(
      (c[obs_i], y[obs_i], t[obs_i]),
      a[obs_i],
      (c[missing_i], y[missing_i], t[missing_i]))

  def get_imputed_values(imputed_probs):
    vals = []
    for i in range(len(missing_i)):
      w = np.random.choice([0, 1], 1, p=a_imputed[i])
      vals.append(w)
    return np.squeeze(np.array(vals))

  resamples = []
  for i in range(k):
    imp_a = a.copy()
    imp_a[missing_i] = get_imputed_values(a_imputed)

    imp_pyac = fit_simple((c, imp_a), y)
    resamples.append(imp_pyac)

  pyac = {key: 0 for key in resamples[0]}
  for key in pyac:
    for resample in resamples:
      pyac[key] += resample[key]
    pyac[key] = pyac[key] / k

  return pyac


def textless(truth, mask):

  c, a, y = truth

  full_pc = fit_bernoulli(c)
  full_pyc = fit_simple(c, y)

  obs_i = [i for i in range(len(mask)) if not mask[i]]
  # cc_pacy = fit_simple((c[obs_i], y[obs_i]), a[obs_i])
  cc_pacy = fit_logis((c[obs_i], y[obs_i]), a[obs_i])

  total_effect = 0
  for a in [0, 1]:
    a_multiplier = (-1, 1)[a]
    a_effect = 0
    for c in [0, 1]:
      # calculate numerator
      pyc_num = full_pyc[(c,)]
      # pacy_num = cc_pacy[(c, 1)]
      inp = np.array([c, 1]).reshape(1, -1)
      pacy_num = logis_proba(cc_pacy, inp)
      if a == 0:
        pacy_num = 1 - pacy_num
      num = pyc_num * pacy_num

      # calculate denominator
      denom = 0
      for y in [0, 1]:
        # pacy_denom = cc_pacy[(c, y)]
        inp = np.array([c, y]).reshape(1, -1)
        pacy_denom = logis_proba(cc_pacy, inp)
        if a == 0:
          pacy_denom = 1 - pacy_denom
        pyc_denom = full_pyc[(c,)]
        if y == 0:
          pyc_denom = 1 - pyc_denom
        denom += pacy_denom * pyc_denom

      # calculate full term for the c summation
      pc = (1 - full_pc, full_pc)[c]
      a_effect += pc * num / denom

    total_effect += a_multiplier * a_effect

  return total_effect


def yelp(n_examples, **kwargs):
  args = synthetic_config.copy()
  args.update(kwargs)

  sampler = YelpData(**args)
  dataset = sampler.load(n_examples)

  c = dataset[:, 0]
  a = dataset[:, 1]
  y = dataset[:, 2]
  t = dataset[:, 3:]
  mask = sampler.mar(a, (c, y, t))

  debug = kwargs.get('debug', False)
  return experiment(c, a, y, t, mask, debug=debug)


def synthetic(n_examples, **kwargs):
  np.random.seed(42)

  config = synthetic_config.copy()
  config.update(kwargs)
  assert config.get('vocab_size') is not None, "Must specify synthetic data vocab_size"

  sampler = SyntheticData(**config)
  c, a, y = sampler.sample_truth(n_examples)
  t = sampler.sample_text((c, a, y))
  mask = sampler.mar(a, (c, y, t))

  debug = kwargs.get('debug', False)
  return experiment(c, a, y, t, mask, debug=debug)


def experiment(c, a, y, t, mask, debug=False):
  obs_i = [i for i in range(len(mask)) if not mask[i]]

  full_pc = fit_bernoulli(c)
  full_pyac = fit_simple((c, a), y)
  cc_pyac = fit_simple((c[obs_i], a[obs_i]), y[obs_i])
  cc_pc = fit_bernoulli(c[obs_i])
  textless_mi_pyac = textless_mi((c, a, y), mask, 20)
  mi_pyac = mi((c, a, y), t, mask, 20)
  bad_mi_pyac = bad_mis((c, a, y), t, mask, 20)

  oracle_err = gformula(full_pyac, full_pc)
  naive_err = gformula(cc_pyac, cc_pc)
  textless_err = gformula(textless_mi_pyac, full_pc)
  bad_mi_err = gformula(bad_mi_pyac, full_pc)
  mi_err = gformula(mi_pyac, full_pc)

  if debug:
    print("\tOracle: {:0.3f}".format(oracle_err))
    print("\tNaive: {:0.6f}".format(naive_err))
    print("\tTextless: {:0.6f}".format(textless_err))
    print("\tbad m.i.: {:0.6f}".format(bad_mi_err))
    print("\tm.i.: {:0.6f}".format(mi_err))

  return [(x - oracle_err) ** 2
          for x in (naive_err, textless_err, bad_mi_err, mi_err)]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("logn_examples", type=float, help="how many examples (log 10)")
  parser.add_argument("--k", type=int, default=10, help="how many duplicate runs to average for this experiment?")
  parser.add_argument("--min_freq", type=int, default=1000,
                      help="min freq for yelp data vocabulary")
  parser.add_argument("--vocab_size", type=int, default=4334,
                      help="vocab size for synthetic data")
  parser.add_argument("--n_vocab", type=int, default=100000,
                      help="how many examples were usedto build the vocab")
  parser.add_argument("--dataset", type=str, default='synthetic')
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--workdir", type=str, default='work/')
  parser.add_argument("--outdir", type=str, default="results/")
  args = parser.parse_args()

  n_examples = int(10 ** args.logn_examples)
  if args.debug:
    print("missing", n_examples, args.k, args.dataset, args.min_freq)

  outfn = "md.{}.{}.{}.{}.json".format(args.dataset, args.logn_examples, args.k, args.min_freq)
  if os.path.exists(os.path.join(args.outdir, outfn)):
    logging.error("{} already exists!".format(outfn))
    return

  job_args = {'debug': args.debug}
  if args.dataset == 'synthetic':
    test_func = synthetic
    job_args['vocab_size'] = args.vocab_size
  elif args.dataset == 'yelp':
    test_func = yelp
    job_args['workdir'] = args.workdir
    job_args['n_vocab'] = args.n_vocab
    job_args['min_freq'] = args.min_freq
  else:
    raise ValueError("unknown dataset {}".format(args.dataset))

  # Run experiments
  results = []
  for i in range(args.k):
    if args.debug:
      print(" {} ".format(i), end='\r')
    try:
      results.append(test_func(n_examples, **job_args))
    except ValueError as e:
      logging.warn("Single run failed with error '{}'".format(e))
      raise e
      pass

  # Compile results
  results = np.array(results)
  means = np.mean(results, axis=0)
  sems = 1.96 * np.std(results, axis=0) / np.sqrt(n_examples)
  models = ['naive', 'textless', "bad_mi", 'mi']

  outlines = []
  for i in range(len(models)):
    outlines.append(json.dumps({
        'model': models[i], 'n': n_examples, 'err': means[i], 'se': sems[i]}))
    if args.debug:
      print("{:8s} {:8.2e} {:8.2e}".format(models[i], means[i], sems[i]))

  # Write to file
  with open(os.path.join(args.outdir, outfn), "w") as outf:
    outf.write("\n".join(outlines))


if __name__ == "__main__":
  main()
