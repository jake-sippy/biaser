# This module is to test if biasing the dataset in a simple way has
# the effect on model performances that we would hope to see.
#
# Reviews at this stage are passed in as lines of json,
# each line is one review of the form:
# {"text": ..., "label": ...}

import utils

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
from multiprocessing import Pool
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        recall_score,
        f1_score
)
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Arugments passed in from command line
global args

# The minimum occurance of words to include as proportion of reviews
MIN_OCCURANCE = 0.05

# The maximum occurance of words to include as proportion of reviews
MAX_OCCURANCE = 1.0

# Ratio to split for the train set (including dev)
TRAIN_SIZE = 0.8

# Should these runs be outputted in log files
LOGGING_ENABLED = True

# Default log path
LOG_PATH = os.path.join('logs', 'bias_test')

# How big to make the process pool
POOL_SIZE = 6

# Mapping of model names to model types
MODELS = {
        'MLP': MLPClassifier(),
        'Linear': LogisticRegression(),
        'RF': RandomForestClassifier(n_estimators=50)
}


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
            'model',
            type=str,
            metavar='MODEL',
            help='The model type, one of: {}'.format(list(MODELS.keys())))
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
            default=LOG_PATH,
            help='Directory to save JSON log files ' \
                 '(default = {})'.format(LOG_PATH))
    parser.add_argument(
            '--quiet',
            action='store_true',
            help='Do not print out any information while running')
    return parser


def run_seed(seed):
    runlog = {}
    if args.quiet:
        sys.stdout = open(os.devnull, 'w')

    # Setting seed #############################################################
    print('\nRunning SEED = {} ------------------------------'.format(seed))
    np.random.seed(seed)
    runlog['seed'] = seed

    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = utils.get_dataset(args.dataset, TRAIN_SIZE, runlog)
#
    # Vectorizing dataset #####################################################
    X_train,  \
    X_test,   \
    y_train,  \
    y_test,   \
    pipeline, \
    feature_names = utils.vectorize_dataset(
            reviews_train,
            reviews_test,
            labels_train,
            labels_test,
            MIN_OCCURANCE,
            MAX_OCCURANCE,
            runlog
    )

    # Resampling dataset #######################################################
    train_df = utils.resample(X_train, y_train, feature_names)

    # Randomly creating bias ###################################################
    bias_idx, bias_word = utils.create_bias(train_df, feature_names, runlog)

    # Training unbiased and biased model #######################################
    model_orig, model_bias = utils.train_models(args.model, MODELS, train_df, runlog)

    # Evaluate both models on biased region R and ~R ###########################
    test_df = pd.DataFrame(data=X_test, columns=None)
    test_df['label'] = y_test
    utils.evaluate_models(model_orig, model_bias, test_df, bias_idx, runlog)

    # Save log #################################################################
    utils.save_log(args.log_dir, runlog)


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    assert (args.seed_low < args.seed_high), 'No seeds in range'
    # Pool(POOL_SIZE).map(run_seed, range(args.seed_low, args.seed_high))
    run_seed(0)
