import itertools
import numpy as np

import sklearn.linear_model


def maybe_stack(inp):
  if type(inp) in [list, tuple]:
    if any(len(x.shape) == 1 for x in inp):
      inp = [x.reshape(-1, 1) if len(x.shape) == 1 else x for x in inp]
      return np.concatenate(inp, axis=1)

    return np.concatenate(inp, axis=1)

  if len(inp.shape) == 1:
    return inp.reshape(-1, 1)
  return inp


def naive(truth):
  v = np.stack(truth, axis=0)

  tot_effect = 0
  for a in [0, 1]:
    where = (v[1, :] == a)
    y_true = np.sum(v[:, np.where(where)][2, :])
    y_tot = np.sum(where)
    tot_effect += (-1, 1)[a] * (y_true / y_tot)

  return tot_effect


def fit_bernoulli(v):
  assert len(v.shape) == 1 or v.shape[1] == 1
  return np.sum(v) / v.shape[0]


def fit_simple(inp, out):
  '''
    inp is a n x k matrix of features
    out is a n x 1 matrix of targets
  '''
  inp = maybe_stack(inp)
  n = inp.shape[0]
  assert out.shape[0] == n
  k = inp.shape[1]

  dist = {}
  assns = list(itertools.product(*[range(2) for _ in range(k)]))
  for assn in assns:
    where = [inp[:, i] == assn[i] for i in range(len(assn))]
    where = np.all(np.stack(where, axis=0), axis=0)
    y_true = np.sum(out[where])
    y_tot = np.sum(where)
    dist[tuple(assn)] = y_true / max(1, y_tot)

  return dist


def fit_logis(inp, out):
  '''
    inp is a n x k matrix of features
    out is a n x 1 matrix of targets
  '''
  inp = maybe_stack(inp)
  n = inp.shape[0]
  assert out.shape[0] == n

  model = sklearn.linear_model.LogisticRegression(C=1e8)
  model.fit(inp, out)
  return model


def logis_proba(model, inp):
  '''
    inp is a n x k matrix of features
  '''
  inp = maybe_stack(inp)
  return model.predict_proba(inp).reshape(-1)[1]


def gformula(pyac, pc):
  tot_effect = 0
  for a in [0, 1]:
    a_multiplier = (-1, 1)[a]
    for c in [0, 1]:
      my_pc = (1 - pc, pc)[c]
      tot_effect += a_multiplier * pyac[(c, a)] * my_pc
  return tot_effect


def mcar(truth, prob):
  n = truth.shape[0]
  return np.random.binomial(1, prob, n)


def impute(features, targets, test_features, debug=False):
  model = fit_logis(features, targets)
  if debug:
    print("acc:", model.score(features, targets))
  test_features = maybe_stack(test_features)
  return model.predict_proba(test_features)
