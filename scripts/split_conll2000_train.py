import os
from collections import defaultdict
import math
import argparse
import random
import numbers
from typing import Dict, List

# One sentence from the NP chunked WSJ corpus split
# Each line contains the current word, the part-of-speech tag as derived by the Brill tagger,
# and its chunk tag as derived from the WSJ corpus
Sentence = List[List[str]]

def write_sentence(sentence: Sentence, handle: os.PathLike) -> None:
    for line in sentence:
        handle.write(line + "\n")
    handle.write("\n")

def read_sentences(path: os.PathLike) -> Dict[int, Sentence]:
    sentences = defaultdict(list)
    current_sentence = 0
    with open(path, 'r') as handle:
        for read_line in handle.readlines():
            line = read_line.strip()
            if not line:
                continue
            sentences[current_sentence].append(line)
            if '. . O' in line:
                current_sentence += 1
    return sentences


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "split",
        type=float,
        default=0.1,
        help="Percentage of the train set to use as the validation set"
    )
    args = parser.parse_args()

    if not (isinstance(args.split, numbers.Real) and args.split >= 0 and args.split <= 1):
        raise ValueError("The percentage of the train set to use as the validation set must be in the range [0,1]")

    train_path = os.path.abspath(os.path.join(__file__, '../../datasets/conll2000/train.txt'))
    dev_path = os.path.abspath(os.path.join(__file__, '../../datasets/conll2000/dev.txt'))
    out_train_path = os.path.abspath(os.path.join(__file__, '../../datasets/conll2000/train_train.txt'))
    old_train_path = os.path.abspath(os.path.join(__file__, '../../datasets/conll2000/train_original.txt'))

    sentences = read_sentences(train_path)
    train_cutoff = math.floor(len(sentences) * (1 - args.split))

    shuffled_sentence_ids = list(sentences.keys())
    random.shuffle(shuffled_sentence_ids)

    with open(out_train_path, 'w+') as train_handle:
        with open(dev_path, 'w+') as dev_handle:
            for num, sentence_id in enumerate(shuffled_sentence_ids):
                if num < train_cutoff:
                    write_sentence(sentences[sentence_id], train_handle)
                else:
                    write_sentence(sentences[sentence_id], dev_handle)

    os.rename(train_path, old_train_path)
    os.rename(out_train_path, train_path)
