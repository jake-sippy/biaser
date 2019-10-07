import os
import gzip
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# This module opens a gzipped Amazon review dataset from
# http://jmcauley.ucsd.edu/data/amazon/ and cleans it to be a json containing
# only the data we need and saving it as a simple csv. In the original json the
# pairs we keep are called "reviewText" and "overall".

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=('Convert an Amazon review dataset from .json.gz to a'
                         ' more managable csv with only text and ac class.'
                         ' datasets at: http://jmcauley.ucsd.edu/data/amazon/')
    )
    parser.add_argument(
            'input',
            type=str,
            metavar='INPUT',
            help='Path of .json.gz file'
    )
    parser.add_argument(
            '--dir',
            type=str,
            metavar='DIR',
            default='clean_data',
            help='Directory to save .json files (default = ./clean_data)'
    )
    args = parser.parse_args()

    print('Reading in dataset: {}'.format(args.input))
    gz = gzip.open(args.input, 'r')
    dataset = pd.read_json(gz, lines=True)

    print('Converting 5-star ratings to binary classification')
    #   < 3 stars => negative (0)
    #   = 3 stars => removed from dataset
    #   > 3 stars => positive (1)
    dataset = dataset[dataset['overall'] != 3]
    dataset['overall'] = dataset['overall'].map(
            {1: 0, 2: 0, 4: 1, 5: 1}
    )

    # filenames are of the form "reviews_[category]_5.json.gz"
    # here we name the cleaned file "reviews_[category].csv"
    output = os.path.basename(args.input).split('_5')[0]
    output = output + '.csv'
    output_path = os.path.join(args.dir, output)
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    print('Writing cleaned dataset to: {}'.format(output_path))
    dataset.to_csv(output_path, header=False, index=False,
            columns=['reviewText', 'overall'])
