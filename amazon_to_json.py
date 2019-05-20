import os
import gzip
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def split_data(in_filename,
               train_filename,
               test_filename,
               train_size,
               downsample):
    json_lines = []
    gz = gzip.open(in_filename, 'r')
    df = pd.read_json(gz, lines=True)
    # converting 5-star ratings to binary sentiment classification
    df = df[df['overall'] != 3]
    df['overall'] = df['overall'].map({1: 'negative',
                                       2: 'negative',
                                       4: 'positive',
                                       5: 'positive'})

    train_df, test_df = train_test_split(df, train_size=train_size,
            test_size=1-train_size)

    with open(train_filename, 'w') as train_f:
        for index, row in train_df.iterrows():
            instance = {'text': row['reviewText'],
                        'label': row['overall']}
            train_f.write(json.dumps(instance) + '\n')

    with open(test_filename, 'w') as test_f:
        for index, row in test_df.iterrows():
            instance = {'text': row['reviewText'],
                        'label': row['overall']}
            test_f.write(json.dumps(instance) + '\n')

    if downsample:
        split = os.path.splitext(train_filename)
        downsample_name = split[0] + '_downsample' + split[1]
        train_majority = train_df[train_df['overall'] == 'positive']
        train_minority = train_df[train_df['overall'] == 'negative']
        maj_downsampled = resample(train_majority,
                                   replace=False,
                                   n_samples=len(train_minority))
        train_downsampled = pd.concat([maj_downsampled, train_minority])
        with open(downsample_name, 'w') as down_f:
            for index, row in train_downsampled.iterrows():
                instance = {'text': row['reviewText'],
                            'label': row['overall']}
                down_f.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    downsample = False
    parser = argparse.ArgumentParser(
            description=('Convert an Amazon review dataset from .json.gz to a'
                         ' more managable json for allennlp.\nDatasets '
                         ' at: http://jmcauley.ucsd.edu/data/amazon/ ')
    )
    parser.add_argument(
            'i',
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
    parser.add_argument(
            '--downsample',
            action='store_true'
    )

    args = parser.parse_args()
    basename = os.path.basename(args.i).split('.')[0].split('_5')[0]
    folder = os.path.join(args.dir, basename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    train_path = os.path.join(folder, 'train.json')
    test_path = os.path.join(folder, 'test.json')
    split_data(args.i, train_path, test_path, 0.8, args.downsample)
