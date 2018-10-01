import argparse
import gzip
import json
import logging
import numpy as np
import os
import time

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


downloader = nltk.downloader.Downloader()
for pkg in ['stopwords', 'punkt']:
  if not downloader.is_installed(pkg):
    downloader.download(pkg)
stemmer = SnowballStemmer('english')
stopwords = stopwords.words('english')


class NumpySerializer(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    else:
      return super(NumpySerializer, self).default(obj)


def load_users(workdir, userfn):
  user_usefuls = {}
  with open(userfn) as inf:
    for line in inf:
      user = json.loads(line.rstrip())
      user_id = user['user_id']
      useful = user['useful']
      user_usefuls[user_id] = int(useful > 1)

  return user_usefuls


def get_words(text, vocab=None):
  words = word_tokenize(text.lower())
  words = [word for word in words if word not in stopwords]
  stems = []
  for word in words:
    stems.append(stemmer.stem(word))
  if vocab is None:
    return stems
  else:
    return set([vocab[stem] for stem in stems if stem in vocab])


def build_vocab(workdir, reviewfn, N=1000000, min_freq=10):
  vocab = {}
  start = time.time()

  freqs_fn = os.path.join(workdir, "freqs.{}.gz".format(N))
  vocab_fn = os.path.join(workdir, "vocab.{}.{}.gz".format(N, min_freq))
  if os.path.exists(freqs_fn) and os.path.exists(vocab_fn):
    logging.warn("vocab already exists, skipping")
    return None

  with open(reviewfn, encoding='utf8') as inf:
    for i in range(N):
      if (i + 1) % 10000 == 0:
        logging.warn("{} processed in {:.1f} seconds".format(i + 1, time.time() - start))
      line = inf.readline()
      review = json.loads(line.rstrip())
      text = review['text']
      words = get_words(text)
      for word in words:
        vocab[word] = vocab.get(word, 0) + 1

  vocab = {key: val for key, val in vocab.items() if val >= min_freq}
  # logging.warn("We found {} words with freq >= {}".format(len(vocab), min_freq))
  with gzip.open(freqs_fn, "wb") as outf:
    outf.write(json.dumps(vocab, cls=NumpySerializer).encode('utf8'))

  indices = {}
  for i, key in enumerate(vocab.keys()):
    indices[key] = i
  vocab_fn = os.path.join(workdir, "vocab.{}.{}.gz".format(N, min_freq))
  with gzip.open(vocab_fn, "wb") as outf:
    outf.write(json.dumps(indices, cls=NumpySerializer).encode('utf8'))

  return vocab


def load_vocab(workdir, N, min_freq):
  vocab_fn = os.path.join(workdir, "vocab.{}.{}.gz".format(N, min_freq))
  with gzip.open(vocab_fn, "rb") as inf:
    vocab = json.loads("".join(x.decode() for x in inf))
  return vocab


def build_dataset(workdir, reviewfn, user_usefuls,
                  total, vocab=None, outfn=None):
  reviews = []

  total = int(total)
  dump_every = int(1e5)
  i = 0
  start = time.time()
  with open(reviewfn, 'r', encoding='utf8') as inf:
    while i < total:
      line = inf.readline()
      if (i + 1) % dump_every == 0:
        logging.warn("{} processed in {}".format(i + 1, time.time() - start))
      review = json.loads(line.rstrip())

      # Treatment: review rating
      stars = review['stars']
      if stars == 3:
        continue
      i += 1
      positive = int(stars > 3)

      # Outcome: received useful
      received_useful = int(review['useful'] > 0)

      # Confounder: user has received usefuls
      user = review['user_id']
      user_is_useful = user_usefuls[user]

      if vocab is None:
        reviews.append(np.array((user_is_useful, positive, received_useful)))
      else:
        # text
        text = review['text']
        words = get_words(text, vocab)
        text_arr = [1 if x in words else 0 for x in range(len(vocab))]
        row = [user_is_useful, positive, received_useful, ]
        reviews.append(np.array(row + text_arr))

      # Every "dump_every" examples, write to file
      if outfn is not None:
        if (i + 1) % dump_every == 0 or i == total:
          logging.warn("dumping!")
          part = (i + 1) // dump_every
          outfn = os.path.join(workdir, "{}.{}.gz".format(outfn, part))
          with gzip.open(outfn, "wt") as outf:
            for review in reviews:
              outf.write("{}\n".format(json.dumps(review, cls=NumpySerializer)))
          reviews = []


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("n_vocab", help="How many reviews to use for building vocab",
                      type=int)
  parser.add_argument("min_freq", help="Minimum word frequence to include in vocab",
                      type=int)
  parser.add_argument("--n_total", type=int, default=None,
                      help="how many examples to write. If None, set equal to n_vocab")
  parser.add_argument("--workdir", type=str, default="work/", help=" ".join((
      "Where are review.json and user.json,",
      "and where should we put vocab and data files?")))
  parser.add_argument("--reviewfn", type=str, default="review.json",
                      help="filename for review file in workdir")
  parser.add_argument("--userfn", type=str, default="user.json",
                      help="filename for user file in workdir")
  args = parser.parse_args()

  if args.n_total is None:
    n_total = args.n_vocab
  else:
    n_total = args.n_total

  if not os.path.exists(args.workdir):
    logging.error("Can't find workdir at {}".format(args.workdir))
    return

  # Locate the review/user files in workdir or via absolute path
  if os.path.exists(args.reviewfn):
    reviewfn = args.reviewfn
  elif os.path.exists(os.path.join(args.workdir, args.reviewfn)):
    reviewfn = os.path.join(args.workdir, args.reviewfn)
  else:
    logging.error("Can't find review file at {}".format(reviewfn))
    return
  if os.path.exists(args.userfn):
    userfn = args.userfn
  elif os.path.exists(os.path.join(args.workdir, args.userfn)):
    userfn = os.path.join(args.workdir, args.userfn)
  else:
    logging.error("Can't find user file at {}".format(userfn))
    return

  # Build/load vocab using n_vocab examples
  vocab = build_vocab(args.workdir, reviewfn,
                      min_freq=args.min_freq, N=args.n_vocab)
  if vocab is None:
    vocab = load_vocab(args.workdir, args.n_vocab, args.min_freq)

  # Load the user-level data
  user_usefuls = load_users(args.workdir, userfn)

  # Use vocab and user data to build dataset
  build_dataset(args.workdir, reviewfn,
                user_usefuls, total=n_total, vocab=vocab,
                outfn='yelpdata.{}.{}'.format(args.n_vocab, args.min_freq))


if __name__ == "__main__":
  main()
