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
# http://jmcauley.ucsd.edu/data/amazon/ and cleans it to be a json containing only
# the data we need. In the original json the pairs we keep are called
# "reviewText" and "overall", which are converted to "text" and "label"
# respectively.

def clean_amazon_reviews(in_filename, out_filename):
    json_lines = []
    gz = gzip.open(in_filename, 'r')
    df = pd.read_json(gz, lines=True)

    # converting 5-star ratings to binary sentiment classification
    # < 3 stars => negative
    # = 3 stars => -removed from dataset-
    # > 3 stars => positive
    df = df[df['overall'] != 3]
    df['overall'] = df['overall'].map(
            {1: 'negative',
             2: 'negative',
             4: 'positive',
             5: 'positive'}
    )

    with open(out_filename, 'w') as out_f:
        print('Writing cleaned json:')
        for index, row in tqdm(df.iterrows()):
            instance = {'text': row['reviewText'],
                        'label': row['overall']}
            out_f.write(json.dumps(instance) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=('Convert an Amazon review dataset from .json.gz to a'
                         ' more managable json for allennlp.\nDatasets '
                         ' at: http://jmcauley.ucsd.edu/data/amazon/ ')
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
            default='data',
            help='Directory to save .json files (default = ./data)'
    )

    args = parser.parse_args()
    # filenames are of the form "reviews_[category]_5.json.gz"
    # here we name the directory "reviews_[category]"
    basename = os.path.basename(args.input).split('_5')[0]
    folder = os.path.join(args.dir, basename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_path = os.path.join(folder, 'reviews.json')
    clean_amazon_reviews(args.input, output_path)
