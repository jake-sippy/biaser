# This module is to test if biasing the dataset in a simple way has
# the effect on model performances that we would hope to see.
#
# Reviews at this stage are passed in as lines of json,
# each line is one review of the form:
# {"text": ..., "label": ...}

import os
import time
import json
import pprint
import sklearn
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

# Range of seeds to loop over
SEED_RANGE = range(0, 100)

# The minimum occurance of words to include as proportion of reviews
MIN_OCCURANCE = 0.05

# The maximum occurance of words to include as proportion of reviews
MAX_OCCURANCE = 1.0

# Directory which holds the datasets
DATA_DIR = 'data'

# Name of dataset to test on
DATASET_NAME = 'reviews_Musical_Instruments'

# Ratio to split for the train set (including dev)
TRAIN_SIZE = 0.8

# Directory to save run logs to
LOG_DIR = 'run_logs'

LOG = False


def load_dataset(path):
    print('Loading dataset from: {} ...'.format(path))

    with open(path, 'rb') as f:
        lines = f.readlines()
        reviews = []
        labels = []

        for line in tqdm(lines):
            json_line = json.loads(line)
            reviews.append(json_line['text'])
            labels.append(1 if json_line['label'] == 'positive' else 0)

    return reviews, labels


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

    print('\t             R      !R')
    print('\torig model | {0:3.2f} | {1:3.2f}'.format(orig_r, orig_not_r))
    print('\tbias model | {0:3.2f} | {1:3.2f}'.format(bias_r, bias_not_r))
    return [[orig_r, orig_not_r], [bias_r, bias_not_r]]


if __name__ == '__main__':
    print('Running bias test...')
    data_path = os.path.join(DATA_DIR, DATASET_NAME, 'reviews.json')
    reviews, labels = load_dataset(data_path)

    for seed in SEED_RANGE:
        runlog = {}

        # Setting seed #########################################################
        print('\nRunning SEED = {} --------------------------------'.format(seed))
        np.random.seed(seed)
        runlog['seed'] = seed
        runlog['dataset'] = DATASET_NAME

        # Splitting dataset ####################################################
        print('Splitting dataset...')
        print('\tTRAIN_SIZE = {}'.format(TRAIN_SIZE))
        reviews_train, \
        reviews_test,  \
        labels_train, \
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
        y_train_orig = train_df['label'].values
        y_train_bias = train_df['label_bias'].values
        X_train = train_df.drop(['label', 'label_bias'], axis=1).values

        model_orig = RandomForestClassifier(n_estimators=100)
        model_bias = RandomForestClassifier(n_estimators=100)
        runlog['model_type'] = 'RandomForestClassifier(n_estimators=100)'

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

        if LOG:
            if not os.path.exists(LOG_DIR):
                os.makedirs(LOG_DIR)
            log_path = os.path.join(LOG_DIR, str(int(time.time())) + '.json')

            print('Writing log to: {}'.format(log_path))
            with open(log_path, 'w') as f:
                json.dump(runlog, f)
