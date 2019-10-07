# This module is to test if biasing the dataset in a simple way has
# the effect on model performances that we would hope to see.
#
# Reviews at this stage are passed in as lines of json,
# each line is one review of the form:
# {"text": ..., "label": ...}
#
# JUST A HEADS UP: This code is pretty messy once you get to main, it still
# needs to be broken up into more readable methods.

import os
import sys
import time
import json
import pprint
import sklearn
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        recall_score,
        f1_score
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# The minimum occurance of words to include as proportion of reviews
MIN_OCCURANCE = 0.05

# The maximum occurance of words to include as proportion of reviews
MAX_OCCURANCE = 1.0

# Ratio to split for the train set (including dev)
TRAIN_SIZE = 0.8

# Should these runs be outputted in log files
LOGGING_ENABLED = True

def setup_argparse():
    parser = argparse.ArgumentParser(
            description=('This script is meant to show that we can reliably ' \
                         'introduce bias in a dataset, and the model that we '\
                         'train on this dataset.'))
    parser.add_argument(
            'dataset',
            type=str,
            metavar='DATASET',
            help='The CSV dataset to bias')
    parser.add_argument(
            'seed_low',
            type=int,
            metavar='SEED_LOW',
            help='The lower bound of seeds to loop over (inclusive)')
    parser.add_argument(
            'seed_high',
            type=int,
            metavar='SEED_HIGH',
            help='The higher bound of seeds to loop over (exclusive)')
    parser.add_argument(
            '--log-dir',
            type=str,
            metavar='LOG_DIR',
            default='run_logs',
            help='Directory to save JSON log files ' \
                 '(default = run_logs/)')
    parser.add_argument(
            '--verbose',
            action='store_true',
            help='Print out information while running')
    return parser


def oversample(df):
    counts = df.label.value_counts()
    smaller_class = df[ df['label'] == counts.idxmin() ]
    larger_class = df[ df['label'] == counts.idxmax() ]
    over = smaller_class.sample(counts.max(), replace=True)
    return pd.concat([over, larger_class], axis=0)


def evaluate_models(model_orig, model_bias, r, not_r):
    X_r = r.drop('label', axis=1).values
    y_r = r['label'].values

    X_not_r = not_r.drop('label', axis=1).values
    y_not_r = not_r['label'].values

    pred_orig_r = model_orig.predict(X_r)
    pred_orig_not_r = model_orig.predict(X_not_r)

    pred_bias_r = model_bias.predict(X_r)
    pred_bias_not_r = model_bias.predict(X_not_r)

    orig_r = accuracy_score(y_r, pred_orig_r)
    orig_not_r = accuracy_score(y_not_r, pred_orig_not_r)

    bias_r = accuracy_score(y_r, pred_bias_r)
    bias_not_r = accuracy_score(y_not_r, pred_bias_not_r)

    print('\t              R     !R')
    print('\torig model | {0:3.2f} | {1:3.2f}'.format(orig_r, orig_not_r))
    print('\tbias model | {0:3.2f} | {1:3.2f}'.format(bias_r, bias_not_r))
    return [[orig_r, orig_not_r], [bias_r, bias_not_r]]


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    dataset_name = args.dataset.split('/')[-1].split('.csv')[0]

    if not args.verbose:
        sys.stdout = open(os.devnull, 'w')

    data = pd.read_csv(args.dataset, header=None, names=['reviews', 'labels'])
    reviews = data['reviews'].astype(str).values
    labels = data['labels'].values

    for seed in range(args.seed_low, args.seed_high):
        runlog = {}

        # Setting seed #########################################################
        print('\nRunning SEED = {} ------------------------------'.format(seed))
        np.random.seed(seed)
        runlog['seed'] = seed
        runlog['dataset'] = dataset_name

        # Splitting dataset ####################################################
        print('Splitting dataset...')
        print('\tTRAIN_SIZE = {}'.format(TRAIN_SIZE))
        reviews_train, \
        reviews_test,  \
        labels_train,  \
        labels_test = train_test_split(reviews, labels, train_size=TRAIN_SIZE)
        runlog['train_size'] = TRAIN_SIZE

        # Vectorizing dataset ##################################################
        print('Converting text dataset to vector representation...')
        print('\tMIN_OCCURANCE = {}'.format(MIN_OCCURANCE))
        print('\tMAX_OCCURANCE = {}'.format(MAX_OCCURANCE))
        vectorizer = CountVectorizer(min_df=MIN_OCCURANCE, max_df=MAX_OCCURANCE)
        X_train = vectorizer.fit_transform(reviews_train).toarray()
        y_train = np.array(labels_train)
        feature_names = vectorizer.get_feature_names()
        print('\tFEATURES EXTRACTED = {}'.format(len(feature_names)))
        train_df = pd.DataFrame(data=X_train, columns=feature_names)
        train_df['label'] = y_train
        runlog['min_occur'] = MIN_OCCURANCE
        runlog['max_occur'] = MAX_OCCURANCE

        # Resample to balance training data ####################################
        print('Resampling to correct class imbalance...')

        value_counts = train_df.label.value_counts()
        print('\tORIGINAL BALANCE = ')
        print('\t\tClass_0 = {}\n\t\tClass_1 = {}'
                .format(value_counts[0], value_counts[1]))

        train_df = oversample(train_df)

        value_counts = train_df.label.value_counts()
        print('\tCORRECTED BALANCE = ')
        print('\t\tClass_0 = {}\n\t\tClass_1 = {}'
                .format(value_counts[0], value_counts[1]))

        # Randomly select word to bias #########################################
        print('Randomly selecting word to bias...')
        bias_idx = np.random.randint(len(feature_names))
        print('\tBIAS_WORD = "{}"'.format(feature_names[bias_idx]))

        train_df['label_bias'] = train_df['label']
        mask = train_df.iloc[:, bias_idx] > 0
        train_df.loc[mask, 'label_bias'] = 0

        # Training unbiased and biased model ###################################
        print('Training models...')
        y_train_orig = train_df['label'].values
        y_train_bias = train_df['label_bias'].values
        X_train = train_df.drop(['label', 'label_bias'], axis=1).values

        model_orig = RandomForestClassifier(n_estimators=100)
        model_bias = RandomForestClassifier(n_estimators=100)
        runlog['model_type'] = 'RandomForestClassifier(n_estimators=100)'
        # model_orig = LinearSVC()
        # model_bias = LinearSVC()
        # runlog['model_type'] = 'LinearSVC()'
        print('\tMODEL_TYPE = {}'.format(runlog['model_type']))

        print('Training unbiased model...')
        start = time.time()
        model_orig.fit(X_train, y_train_orig)
        end = time.time()
        print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))

        print('Training biased model...')
        start = time.time()
        model_bias.fit(X_train, y_train_bias)
        end = time.time()
        print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))

        # Evaluate both models on biased region R and ~R ######################
        print('Evaluating unbiased and biased models on test set...')
        X_test = vectorizer.transform(reviews_test).toarray()
        y_test = np.array(labels_test)
        test_df = pd.DataFrame(data=X_test, columns=feature_names)
        test_df['label'] = y_test
        mask = test_df.iloc[:, bias_idx] > 0
        R = test_df[mask]
        not_R = test_df[~mask]

        runlog['results'] = evaluate_models(model_orig, model_bias, R, not_R)

        if LOGGING_ENABLED:
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)
            log_path = os.path.join(args.log_dir,
                    str(int(time.time())) + '.json')

            print('Writing log to: {}'.format(log_path))
            with open(log_path, 'w') as f:
                json.dump(runlog, f)
