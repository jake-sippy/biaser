# This module is to test if biasing the dataset results in a biased model (as we
# would hope). We test this by biasing the dataset, training models on both the
# biased and unbiased versions of the dataset and checking whether the models
# trained on the biased version have significantly altered performance.

import utils
import biases

import os
import sys
import time
import json
import pprint
import sklearn
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot (ssh only)
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
from sklearn.tree import DecisionTreeClassifier, plot_tree

global args                     # Arguments from cmd line
TEST_NAME = 'bias_test'         # Name of this test
LOG_PATH = 'logs'               # Top level directory for log files
POOL_SIZE = 10                  # How many workers to spawn (one per seed)
TRAIN_SIZE = 0.9                # Train split ratio (including dev)
MIN_OCCURANCE = 0.20            # Min occurance for n-grams to be included
MAX_OCCURANCE = 0.50            # Max occurance for n-grams to be included
LOGGING_ENABLED = True          # Save logs for this run if true

# Mapping of model names to model objects
MODELS = {
    # 'mlp': MLPClassifier(),   # cannot use sample weights
    'logistic': LogisticRegression(solver='lbfgs'),
    'rf': RandomForestClassifier(n_estimators=50),
    'dt': DecisionTreeClassifier(max_depth=3)
}


def run_seed(seed):
    """
    Runs a single seed of the test.

    Run a single seed of the test including biasing data, training models, and
    evaluating performance across regions of the dataset.
    """
    if args.quiet: sys.stdout = open(os.devnull, 'w')

    # Set metadata in runlog
    runlog = {}
    runlog['test_name'] = TEST_NAME
    runlog['bias_len'] = args.bias_length
    runlog['seed'] = seed

    print('\nRunning SEED = {} ------------------------------'.format(seed))
    np.random.seed(seed)

    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = utils.load_dataset(args.dataset, TRAIN_SIZE, runlog)

    # Create bias #############################################################
    bias_obj = biases.ComplexBias(
            reviews_train,
            labels_train,
            runlog['bias_len'],
            MIN_OCCURANCE,
            MAX_OCCURANCE,
            runlog
    )
    labels_train_bias, biased_train = bias_obj.bias(reviews_train, labels_train)
    labels_test_bias, biased_test = bias_obj.bias(reviews_test, labels_test)

    # Preprocessing reviews TODO Generalize for pytorch models
    X_train,  \
    X_test,   \
    pipeline, \
    feature_names = utils.vectorize_dataset(
            reviews_train,
            reviews_test,
            MIN_OCCURANCE,
            MAX_OCCURANCE,
            runlog
    )

    # Convert to pandas df
    train_df = pd.DataFrame(data=X_train, columns=feature_names)
    train_df['label_orig'] = labels_train
    train_df['label_bias'] = labels_train_bias
    train_df['biased'] = biased_train

    test_df = pd.DataFrame(data=X_test, columns=feature_names)
    test_df['label_orig'] = labels_test
    test_df['label_bias'] = labels_test_bias
    test_df['biased'] = biased_test

    # Resampling dataset #######################################################
    train_df = utils.resample(train_df, feature_names)

    # Training unbiased and biased model #######################################
    model_orig, model_bias = utils.train_models(
            args.model,
            MODELS,
            train_df,
            runlog
    )

    # Evaluate both models on biased region R and ~R ###########################
    utils.evaluate_models(model_orig, model_bias, test_df, runlog)

    print('Plotting biased decision tree...')
    importances = sorted(
            list(zip(feature_names, model_bias.feature_importances_)),
            key=lambda x: x[1],
            reverse=True
    )
    print('\tBIAS_WORDS = {}'.format(bias_obj.bias_words))
    print('\tIMPORTANCES= {}'.format(importances[:10]))
    fig = plt.gcf()
    fig.set_size_inches(150, 100)
    ax = plt.gca()
    plot_tree(
            decision_tree=model_bias,
            feature_names=feature_names,
            filled=True,
            ax=ax
    )
    filename = 'decision_tree_{}.png'.format(seed)
    print('Saving tree plot to: {}'.format(filename))
    plt.savefig(filename, bbox_inches='tight')


def setup_argparse():
    parser = argparse.ArgumentParser(
            description=('This script is meant to show that we can reliably ' \
                         'introduce bias in a dataset, and the model that we '\
                         'train on this dataset.'))
    parser.add_argument(
            'dataset',
            type=str,
            metavar='DATASET',
            help='CSV dataset to bias')
    parser.add_argument(
            'model',
            type=str,
            metavar='MODEL',
            help='Model type, one of: {}'.format(list(MODELS.keys())))
    parser.add_argument(
            'seed_low',
            type=int,
            metavar='SEED_LOW',
            help='Lower bound of seeds to loop over (inclusive)')
    parser.add_argument(
            'seed_high',
            type=int,
            metavar='SEED_HIGH',
            help='Higher bound of seeds to loop over (exclusive)')
    parser.add_argument(
            'bias_length',
            type=int,
            metavar='BIAS_LEN',
            help='Number of features to include in bias')
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

    # Check args
    args = parser.parse_args()
    assert (args.seed_low < args.seed_high), \
            'No seeds in range [{}, {})'.format(args.seed_low, args.seed_high)
    assert args.model in MODELS, \
            'Model name not recognized ({}), must be one of {}'.format(
                    args.model, list(MODELS.keys()))
    return args


if __name__ == '__main__':
    args = setup_argparse()
    seeds = range(args.seed_low, args.seed_high)
    if POOL_SIZE > 1:
        pool = Pool(POOL_SIZE)
        pool.map(run_seed, seeds)
        pool.close()
        pool.join()
    else:
        for seed in seeds:
            run_seed(seed)
