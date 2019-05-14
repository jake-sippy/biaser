import os
import gzip
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def to_json_lines(in_path):
    json_lines = []
    gz = gzip.open(in_path, 'r')
    for l in tqdm(gz):
        line = eval(l)
        text = str(line['reviewText'])
        label = int(line['overall'])

        # converting 5-star ratings to binary sentiment classification
        if label == 1 or label == 2:
            label = 'negative'
        elif label == 3:
            continue
        elif label == 4 or label == 5:
            label = 'positive'

        json_lines.append(json.dumps({'text' : text, 'label' : label}))
    return json_lines


def split_json(lines, train_filename, test_filename, train_size):
    idx = np.random.permutation(len(lines))
    split = int(len(lines)* train_size)
    train_idx = idx[:split]
    test_idx = idx[split:]

    with open(train_filename, 'w') as train_f:
        for i in train_idx:
            train_f.write(lines[i] + '\n')

    with open(test_filename, 'w') as test_f:
        for i in test_idx:
            test_f.write(lines[i] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=('Convert an Amazon review dataset from .json.gz to a'
                         ' more managable json for allennlp.\nDatasets '
                         ' at: http://jmcauley.ucsd.edu/data/amazon/'))
    parser.add_argument('i', metavar='gz_path', help='Path of .json.gz file')
    parser.add_argument(
            '--train',
            metavar='train_path',
            help='Path to save .json of training split'
    )
    parser.add_argument(
            '--test',
            metavar='test_path',
            help='Path to save .json of testing split'
    )
    args = parser.parse_args()
    if args.train is None:
        args.train = args.i.split('.')[0] + '_train.json'
    if args.test is None:
        args.test = args.i.split('.')[0] + '_test.json'
    json_lines = to_json_lines(args.i)
    split_json(json_lines, args.train, args.test, 0.8)
