import os
import json
import logging
import gzip

import numpy as np

from utils import maybe_stack

synthetic_config = {
    'topic_std': 0.2,
    'missing_bias': 0.7,
    'missing_effect_std': 0.1,
    'missing_effects': (.2, -.4),
}


class YelpData:
  def __init__(self, workdir, n_vocab, min_freq=1000, **kwargs):

    vocabfn = "vocab.{}.{}.gz".format(n_vocab, min_freq)
    if not os.path.exists(os.path.join(workdir, vocabfn)):
      raise IOError("can't find {} to calculate vocab size".format(vocabfn))

    with gzip.open(os.path.join(workdir, vocabfn)) as inf:
      vocab = json.loads(inf.readline().decode())
      self.vocab_size = len(vocab)

    self.workdir = workdir
    self.n_vocab = n_vocab

    self.missing_bias = kwargs['missing_bias']
    self.missing_effect_std = kwargs['missing_effect_std']
    self.missing_effects = (
        *kwargs['missing_effects'],
        *np.random.normal(0, self.missing_effect_std, self.vocab_size))
    self.min_freq = min_freq
    self.dataset = []
    self.max_seen = 0

  def load(self, num_examples, allow_overlap=False, enforce_full=True):

    if len(self.dataset) >= num_examples:
      if allow_overlap:
        return np.array(self.dataset[:num_examples])
      else:
        if len(self.dataset) >= num_examples + self.max_seen:
          return np.array(self.dataset[self.max_seen:self.max_seen + num_examples])

    part = 0
    dataset = []
    total_to_load = num_examples
    if not allow_overlap:
      total_to_load += self.max_seen

    while len(dataset) < total_to_load:
      infn = "yelpdata.{}.{}.{}.gz".format(self.n_vocab, self.min_freq, part)
      if not os.path.exists(os.path.join(self.workdir, infn)):
        logging.error("stopping before part #{}, {} doesn't exist".format(part, infn))
        break
      with gzip.open(os.path.join(self.workdir, infn), "rb") as inf:
        for line in inf:
          dataset.append(json.loads(line.decode()))
          if len(dataset) > total_to_load:
            break

      part += 1

    self.dataset = dataset
    if allow_overlap:
      dataset = np.array(dataset[: num_examples])
    else:
      dataset = np.array(dataset[self.max_seen: self.max_seen + num_examples])

    if len(dataset) < num_examples and enforce_full:
      raise ValueError("Unable to find {} examples for Yelp dataset".format(num_examples))

    self.max_seen += num_examples
    return dataset

  def mar(self, truth, confounds):
    n = truth.shape[0]
    confounds = maybe_stack(confounds)

    prob = np.ones(n) * self.missing_bias + np.dot(confounds, self.missing_effects)
    prob = np.clip(prob, 0.01, 0.99)
    return np.random.binomial(1, prob, n)


class SyntheticData:
  def __init__(self, **kwargs):
    self.c_bias = 0.4
    self.a_bias = 0.40
    self.ca_effect = -0.3
    self.y_bias = 0.5
    self.cy_effect = 0.2
    self.ay_effect = 0.1

    self.vocab_size = kwargs['vocab_size']
    self.topic_std = kwargs['topic_std']
    self.topic_bias = np.ones(self.vocab_size) * 0.5
    self.topic_effects = [np.random.choice([-1., 1.], self.vocab_size) *
                          np.random.normal(0, self.topic_std, self.vocab_size)
                          for _ in range(3)]

    # NOTE This is a way to prevent the measurement error classifier from getting perfect
    #      accuracy on the synthetic data. When vocab size increases, sufficient training
    #      data make it possible to get 100% accuracy. If many "words"
    #      have no association with the treatment, the logistic regression gets harder.
    num_nonzero_effects = max(100, self.vocab_size // 50)
    self.topic_effects_mask = np.concatenate(
        [np.ones(num_nonzero_effects), np.zeros(self.vocab_size - num_nonzero_effects)])
    np.random.shuffle(self.topic_effects_mask)
    self.topic_effects *= self.topic_effects_mask

    self.missing_bias = kwargs['missing_bias']
    self.missing_effect_std = kwargs['missing_effect_std']
    self.missing_effects = (
        *kwargs['missing_effects'],
        *np.random.normal(0, self.missing_effect_std, self.vocab_size))

  def sample_truth(self, n):
    c = np.random.choice([0, 1], n, p=(1 - self.c_bias, self.c_bias))

    a_prob = self.a_bias * np.ones(n) + self.ca_effect * c
    a = np.random.binomial(1, a_prob, n)

    y_prob = self.y_bias * np.ones(n) + self.cy_effect * c + self.ay_effect * a
    y = np.random.binomial(1, y_prob, n)

    return (c, a, y)

  def sample_text(self, truth):
    truth = maybe_stack(truth)
    n = truth.shape[0]
    topic = np.tile(self.topic_bias, n).reshape(n, self.vocab_size)
    for i in range(truth.shape[1]):
      topic += truth[:, i].reshape(n, 1) * np.tile(
          self.topic_effects[i], n).reshape(n, self.vocab_size)
    topic = np.clip(topic, 0.01, 0.99)
    words = []
    for i in range(n):
      word = (np.random.random(self.vocab_size) < topic[i]).astype(np.int32)
      words.append(word)
    words = np.array(words)
    return words

  def mar(self, truth, confounds):
    n = truth.shape[0]
    confounds = maybe_stack(confounds)

    prob = np.ones(n) * self.missing_bias + np.dot(confounds, self.missing_effects)
    prob = np.clip(prob, 0.01, 0.99)
    return np.random.binomial(1, prob, n)
