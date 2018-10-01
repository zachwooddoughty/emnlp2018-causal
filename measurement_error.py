import itertools
import json
import logging
import os
import argparse

import numpy as np
import sklearn.linear_model

from datasets import YelpData, SyntheticData, synthetic_config
from utils import gformula, fit_simple, fit_bernoulli


def get_error_rate(proxy, truth, truth_val, confounds=None, confound_assn=None):
  '''
  Given a proxy (A*) and truth (A), calculate the error rate when A=1.
  If confounds and confound_assn are given, limit this calculation to when
    confounds have the given assignment
  '''
  where = []
  if confounds is not None and confound_assn is not None:
    where = [confounds[i, :] == confound_assn[i] for i in range(len(confound_assn))]

  true_where = where + [truth == truth_val]
  true_where = np.all(np.stack(true_where, axis=0), axis=0)
  true = sum(true_where)

  correct_where = where + [proxy == truth_val, truth == truth_val]
  correct_where = np.all(np.stack(correct_where, axis=0), axis=0)
  correct = sum(correct_where)

  if true == 0:
    return 0
  return 1 - correct / true


def calculate_error_matrix(proxy, truth, confounds=None, debug=False):
  '''
    Given the proxy (A*) and truth (A), calculate the correction matrix
    to adjust the causal effect calculations
  '''
  errs = {}
  if confounds is None:
    confound_assns = [()]
  else:
    confound_assns = list(itertools.product(*[range(2) for _ in confounds]))

  for truth_val in [0, 1]:
    for confound_assn in confound_assns:
      err_rate = get_error_rate(proxy, truth, truth_val, confounds, confound_assn)
      errs[(truth_val, ) + confound_assn] = err_rate

  return errs


def get_dist(dist, debug=False):
  '''
  Calculate the probability mass on all variable assignments
  '''
  n = dist.shape[1]
  d = {}
  for assn in itertools.product(*[range(2) for _ in range(3)]):
    where = [dist[i, :] == assn[i] for i in range(len(assn))]
    where = np.all(np.stack(where, axis=0), axis=0)
    d[tuple(assn)] = np.sum(where) / n

  return d


def get_corrected_dist(dist, errs, truth, proxied_index, confounds=(), debug=False):
  '''
  Given a proxy distribution dist, error estimates, and a held-out truth,
    calculate the new dist that comes from using the error estimates to correct the proxy.
  dist: p(C, A*, Y)
  errs: p(A*, A)
  truth: p(C, A, Y)
  '''
  start = get_dist(dist)

  corrected = {}
  for assn in itertools.product(*[range(2) for _ in range(len(dist) - 1)]):
    assn0 = list(assn[:])
    assn0.insert(proxied_index, 0)
    assn1 = list(assn[:])
    assn1.insert(proxied_index, 1)
    confound_assn = [assn0[i] for i in confounds]

    # error assignment indexing is proxied value first
    err1 = errs[(1, ) + tuple(confound_assn)]
    err0 = errs[(0, ) + tuple(confound_assn)]

    corrected0 = (1 - err1) * start[tuple(assn0)] - err1 * start[tuple(assn1)]
    corrected0 /= (1 - err1 - err0)
    if np.isfinite(corrected0):
      corrected[tuple(assn0)] = corrected0
    else:
      corrected[tuple(assn0)] = start[tuple(assn0)]

    corrected1 = - err0 * start[tuple(assn0)] + (1 - err0) * start[tuple(assn1)]
    corrected1 /= (1 - err1 - err0)
    if np.isfinite(corrected1):
      corrected[tuple(assn1)] = corrected1
    else:
      corrected[tuple(assn1)] = start[tuple(assn1)]

    if debug:
      print("errors: {:0.3f} and {:0.3f}".format(err0, err1))
      print("correcting took:")
      print("p(c'={}, a={}, y={}) from {:0.3f} to {:0.3f}, truth is {:0.3f}".format(
          *(assn0 + [start[tuple(assn0)], corrected0, truth[tuple(assn0)]])))
      print("p(c'={}, a={}, y={}) from {:0.3f} to {:0.3f}, truth is {:0.3f}".format(
          *(assn1 + [start[tuple(assn1)], corrected1, truth[tuple(assn1)]])))

  return corrected


def check_restoration(dist1, dist2):
  '''
  Calculate the L2 distance between two distributions as a sanity check.
  ''' 
  dist_err = 0
  for key in dist1:
    dist_err += (dist1[key] - dist2[key]) ** 2

  return dist_err


def dist_pyac(dist):
  '''
  Assume dist is ordered as c, a, y
  Given a distribution dictionary from get_dist, calculate p(Y=1 | A, C)
  '''
  pyac = {}
  for a in [0, 1]:
    for c in [0, 1]:
      pyac[(c, a)] = dist[(c, a, 1)] / (dist[(c, a, 1)] + dist[(c, a, 0)])

  return pyac


def dist_pc(dist):
  ''' Given a distribution dictionary from get_dist, calculate p(C=1) '''
  pc = 0
  for a in [0, 1]:
    for y in [0, 1]:
      pc += dist[(1, a, y)]

  return pc


def textless_impute(train, test, n, num_train,
                    proxy_i=1, confound_i=(), debug=False):
  '''
  Unused code for measurement error without text.
  In our experiments, always led to singular matrix.
  '''

  model = sklearn.linear_model.LogisticRegression()
  features = np.concatenate((train[:1, :], train[2:3, :]), axis=0)
  model.fit(np.transpose(features[:, :num_train]), train[proxy_i, :num_train])
  if debug:
    print("Model dev acc: {:0.3f}".format(model.score(
        np.transpose(features[:, num_train:]), train[proxy_i, num_train:])))
  train_proxy = model.predict(np.transpose(features[:, num_train:]))

  if confound_i:
    err_confounds = train[confound_i, num_train:]
  else:
    err_confounds = None

  errs = calculate_error_matrix(train_proxy, train[proxy_i, num_train:],
                                confounds=err_confounds, debug=debug)
  if debug:
    print("Dev error rates: {}".format(", ".join(["{:0.1f}".format(err) for err in errs.values()])))

  features = np.concatenate((test[:1, :], test[2:3, :]), axis=0)
  if debug:
    print("Test acc: {:0.3f}".format(model.score(np.transpose(features), test[proxy_i, :])))
  proxy = test[:3, :].copy()
  proxy[proxy_i, :] = model.predict(np.transpose(features))
  truth = test[:3, :]

  start_err = check_restoration(get_dist(truth), get_dist(proxy))
  new_dist = get_corrected_dist(proxy, errs, get_dist(truth), proxy_i, confound_i, debug)
  end_err = check_restoration(get_dist(truth), new_dist)

  if debug:
    print("true dist")
    print(truth.shape)
    get_dist(truth, debug)
    print("proxy dist")
    get_dist(proxy, debug)
    print("start err: {:0.3e}".format(start_err))
    print("end err: {:0.3e}".format(end_err))
    print("proxy to correction diff: {:0.3f}".format(
          check_restoration(new_dist, get_dist(proxy))))
    print("Correction reduced error by a factor of {:0.1f}".format(start_err / end_err))

  return new_dist


def impute_and_correct(train, test,  num_train,
                       proxy_i=1, confound_i=(), debug=False):
  '''
  train: training data
  test: testing data
  num_train: how many examples of training data to use for training,
    (leaving the rest for development)
  proxy_i: what is the index of the proxied variable (e.g. 1)
  confound_i: what are the indices of the proxy's confounders (e.g. 0, 2)
  '''

  # features are everything but the proxy variable
  train_features = np.concatenate((train[:proxy_i, :], train[1 + proxy_i:, :]), axis=0)

  model = sklearn.linear_model.LogisticRegression()
  model.fit(np.transpose(train_features[:, :num_train]), train[proxy_i, :num_train])
  train_proxy = model.predict(np.transpose(train_features[:, num_train:]))

  if confound_i:
    err_confounds = train[confound_i, num_train:]
  else:
    err_confounds = None

  errs = calculate_error_matrix(train_proxy, train[proxy_i, num_train:],
                                confounds=err_confounds, debug=debug)

  # features are everything but the proxy variable, including text
  test_features = np.concatenate((test[:proxy_i, :], test[1 + proxy_i:, :]), axis=0)

  # "proxy" and "truth" arrays have all non-text variables
  nontext_vars = sorted(confound_i + (proxy_i, ))
  proxy = test.copy()
  proxy[proxy_i, :] = model.predict(np.transpose(test_features))
  proxy = proxy[nontext_vars, :]
  truth = test[nontext_vars, :]

  start_err = check_restoration(get_dist(truth), get_dist(proxy))

  new_dist = get_corrected_dist(proxy, errs, get_dist(truth), proxy_i, confound_i, debug)
  end_err = check_restoration(get_dist(truth), new_dist)

  if debug:
    print("Model dev acc: {:0.3f}".format(model.score(
        np.transpose(train_features[:, num_train:]), train[proxy_i, num_train:])))
    print("Dev error rates: {}".format(", ".join(["{:0.1f}".format(err) for err in errs.values()])))
    print("Test acc: {:0.3f}".format(model.score(np.transpose(test_features), test[proxy_i, :])))
    print("true dist")
    print(get_dist(truth, debug))
    print("proxy dist")
    print(get_dist(proxy, debug))
    print("start err: {:0.3e}".format(start_err))
    print("end err: {:0.3e}".format(end_err))
    print("proxy to correction diff: {:0.3f}".format(
        check_restoration(new_dist, get_dist(proxy))))
    print("Correction reduced error by a factor of {:0.1f}".format(start_err / end_err))

  return new_dist, proxy


def train_adjust(train, test, proxy_i=1, confound_i=(), debug=False):
  '''
  Given train and test data, train a logistic regression classifier to
    impute a proxy for the missing variables, then calculate the errors
    from an oracle in causal effect estimation.
  '''
  n = test.shape[0]

  # use half the train set for training, half for dev and error calculation
  num_train = train.shape[1] // 2

  truth = test[:3, :]
  new_dist, proxy = impute_and_correct(train, test, n, num_train,
                                       proxy_i, confound_i, debug)

  oracle_effect = gformula(dist_pyac(get_dist(truth)), dist_pc(get_dist(truth)))

  # Instead of training our model for the mismeasurement, just report
  #   the causal effect present in the training dataset
  naive_effect = gformula(fit_simple(
      np.transpose(train[:2, :]), train[2, :]),
      fit_bernoulli(train[0, :]))

  misspecified_effect = gformula(dist_pyac(get_dist(proxy)), dist_pc(get_dist(proxy)))
  corrected_effect = gformula(dist_pyac(new_dist), dist_pc(new_dist))

  if debug:
    print("True dist gives effect: {:0.3f}".format(oracle_effect))
    print("Naive approach gives effect: {:0.3f}".format(naive_effect))
    print("Misspecified dist gives effect: {:0.3f}".format(misspecified_effect))
    print("corrected dist gives effect: {:0.3f}".format(corrected_effect))

  return [(x - oracle_effect) ** 2
          for x in (naive_effect, misspecified_effect, corrected_effect)]


def synthetic(n_examples, n_train, **kwargs):
  '''
  Run a synthetic experiment using n_examples examples in target dataset p(A*, C, Y)
  and n_train examples of external data to estimate p(A*, A)
  '''
  np.random.seed(42)

  proxy_i = 1
  confound_i = (0, 2, )

  config = synthetic_config.copy()
  config.update(kwargs)
  assert config.get('vocab_size') is not None, "Must specify synthetic data vocab_size"

  sampler = SyntheticData(**config)

  truth = sampler.sample_truth(n_examples)
  truth_t = sampler.sample_text(truth[proxy_i])
  external = sampler.sample_truth(n_train)
  external_t = sampler.sample_text(external[proxy_i])

  debug = kwargs.get('debug', False)

  return train_adjust(
      np.concatenate((np.array(external), np.transpose(external_t)), axis=0),
      np.concatenate((np.array(truth), np.transpose(truth_t)), axis=0),
      proxy_i, confound_i, debug=debug)


def yelp(n_examples, n_train, **kwargs):
  '''
  Run a Yelp experiment using n_examples examples in target dataset p(A*, C, Y)
  and n_train examples of external data to estimate p(A*, A)
  '''
  proxy_i = 1
  confound_i = (0, 2, )

  args = synthetic_config.copy()
  args.update(kwargs)

  sampler = YelpData(**args)
  truth = sampler.load(n_examples)
  external = sampler.load(n_train)

  debug = kwargs.get('debug', False)

  return train_adjust(
      np.transpose(external), np.transpose(truth),
      proxy_i, confound_i, debug=debug)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("logn_examples", type=float, help="how many examples (log 10)")
  parser.add_argument("--k", type=int, default=10, help="how many runs for each?")
  parser.add_argument("--dataset", type=str, default='synthetic')
  parser.add_argument("--min_freq", type=int, default=10,
                      help="min freq for yelp data vocabulary")
  parser.add_argument("--vocab_size", type=int, default=4334,
                      help="vocab size for synthetic data")
  parser.add_argument("--n_vocab", type=int, default=100000,
                      help="how many examples to use to build vocab")
  parser.add_argument("--debug", type=bool, default=False)
  parser.add_argument("--workdir", type=str, default='work/')
  parser.add_argument("--outdir", type=str, default="results/")
  args = parser.parse_args()

  n_examples = int(10 ** args.logn_examples)
  n_train = max(1000, n_examples // 100)
  if args.debug:
    print("measure", n_examples, n_train, args.k, args.dataset, args.min_freq)

  outfn = "me.{}.{}.{}.{}.json".format(args.dataset, args.logn_examples, args.k, args.min_freq)
  if not os.path.exists(args.outdir):
    logging.error("{} doesn't exist, quitting".format(args.outdir))
    return
  if os.path.exists(os.path.join(args.outdir, outfn)):
    logging.error("{} already exists, quitting".format(outfn))
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

  results = []
  for i in range(args.k):
    if args.debug:
      print(" {} ".format(i), end='\r')
    results.append(test_func(n_examples, n_train, **job_args))

  results = np.array(results)
  means = np.nanmean(results, axis=0)
  sems = 1.96 * np.nanstd(results, axis=0) / np.sqrt(n_examples)
  models = ['naive', "misspecified", 'correct']

  if args.debug:
    print("Missp NaNs: {} of {}".format(sum(np.isnan(results[:, 1])), args.k))
    print("Correct NaNs: {} of {}".format(sum(np.isnan(results[:, 2])), args.k))

  outlines = []
  for i in range(len(models)):
    outlines.append(json.dumps({
        'model': models[i], 'n': n_examples, 'err': means[i], 'se': sems[i]}))
    if args.debug:
      print("{:8s} {:8.2e} {:8.2e}".format(models[i], means[i], sems[i]))

  with open(os.path.join(args.outdir, outfn), "w") as outf:
    outf.write("\n".join(outlines))


if __name__ == "__main__":
  main()
