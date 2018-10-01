This is the code for the experiments in the EMNLP '18 paper, [Challenges of Using Text Classifiers for Causal Inference](http://arxiv.org).

To get the raw data, either:
 - download latest Yelp dataset from https://www.yelp.com/dataset/download, or:
 - email zach@cs.jhu.edu to get the exact data used for the paper.

To preprocess the data, run ```python build_yelp_dataset.py n_vocab min_freq --workdir raw_data_directory/```
 - ```n_vocab``` is the number of reviews to load in creating the word vocabulary.
 - ```min_freq``` is the minimum frequency a word can occur in those ```n_vocab``` reviews to be included.
 - the flag ```--n_total``` controls how many actual reviews to preprocess, and by default equals ```n_vocab```.
 - ```n_vocab``` and ```min_freq``` are the two params that specify a dataset in all Yelp experiments.
 
To run a Yelp missing data experiment, run:

  ```python missing_data.py logn_examples --dataset yelp --n_vocab n_vocab --min_freq min_freq --workdir raw_data_directory/```

To run a synthetic missing data experiment, run:

  ```python missing_data.py logn_examples --dataset synthetic --vocab_size vocab_size```
  
To run a Yelp measurement error experiment, run:

  ```python measurement_error.py logn_examples --dataset yelp --n_vocab n_vocab --min_freq min_freq --workdir raw_data_directory/```
  
To run a synthetic measurement error experiment, run:

  ```python measurement_error.py logn_examples --dataset synthetic --vocab_size vocab_size```
  
If you use this code, please cite:

```
@inproceedings{wooddoughty2018challenges,
  title={Challenges of Using Text Classifiers for Causal Inference},
  author={Wood-Doughty, Zach, Ilya Shpitser, and Mark Dredze},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year={2018}
}
```
