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

SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def split_data(in_filename,
               train_filename,
               dev_filename,
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

    train_df, test_df = train_test_split(df,
                                         train_size=train_size,
                                         test_size=1-train_size)
    train_df, dev_df = train_test_split(train_df,
                                        train_size=train_size,
                                        test_size=1-train_size)

    if downsample:
        # TODO make finding the majority class automatic
        train_majority = train_df[train_df['overall'] == 'positive']
        train_minority = train_df[train_df['overall'] == 'negative']
        maj_downsampled = resample(train_majority,
                                   replace=False,
                                   n_samples=len(train_minority))
        train_df = pd.concat([maj_downsampled, train_minority])

    with open(train_filename, 'w') as train_f:
        print('Writing train split:')
        for index, row in tqdm(train_df.iterrows()):
            instance = {'text': row['reviewText'],
                        'label': row['overall']}
            train_f.write(json.dumps(instance) + '\n')

    with open(dev_filename, 'w') as dev_f:
        print('Writing dev split:')
        for index, row in tqdm(dev_df.iterrows()):
            instance = {'text': row['reviewText'],
                        'label': row['overall']}
            dev_f.write(json.dumps(instance) + '\n')

    with open(test_filename, 'w') as test_f:
        print('Writing test split:')
        for index, row in tqdm(test_df.iterrows()):
            instance = {'text': row['reviewText'],
                        'label': row['overall']}
            test_f.write(json.dumps(instance) + '\n')


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
    parser.add_argument(
            '--dont-downsample',
            action='store_true'
    )

    args = parser.parse_args()
    basename = os.path.basename(args.input).split('.')[0].split('_5')[0]
    folder = os.path.join(args.dir, basename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    train_path = os.path.join(folder, 'train.json')
    dev_path = os.path.join(folder, 'dev.json')
    test_path = os.path.join(folder, 'test.json')
    split_data(args.input, train_path, dev_path, test_path, 0.8, not args.dont_downsample)
