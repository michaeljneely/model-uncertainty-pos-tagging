# Deep Learning for Natural Language Processing - POS Tagging

University of Amsterdam Deep Learning for Natural Language Processing Fall 2020 Mini Project - POS Tagging

## Abstract

TODO

## Authors

- Leila F.C. Talha
- Michael J. Neely
- Stefan F. Schouten

## Setup

Prepare a Python virtual environment and install the necessary packages.

```shell
python3 -m venv v-dl4nlp-pos-tagging
source v-dl4nlp-pos-tagging/bin/activate
pip install torch
pip install -r requirements.txt
python -m spacy download en
```

## Datasets

1. [CoNLL-2000](https://www.clips.uantwerpen.be/conll2000/chunking/)

    Download the train and tests sets to the `datasets/conll200` directory and run the `scripts/split_conll2000_train.py` script.
    Provide the percentage of the train set to use as the validation set with a positional argument. Default: 0.1

## Running the Experiment

Train the Meta-BiLSTM morphosyntactic tagger, calculate its uncertainty on the test set, and generate some interesting figures by running:

```shell
allennlp uncertainty-experiment experiments/conll2000_meta_tagger_separate_mcdrop.jsonnet
```

By default, generated artifacts are saved in the `outputs/` directory.
