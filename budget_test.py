import utils

import os
import sys
import time
import json
import sklearn
import argparse
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import binom

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
        classification_report,
        mean_squared_error,
        accuracy_score,
        recall_score,
        f1_score
)
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from explainers import(
        LimeExplainer,
        ShapExplainer,
        ShapZerosExplainer,
        ShapMedianExplainer,
        GreedyExplainer,
        LogisticExplainer,
        RandomExplainer,
)


global args                 # Arguments passed in from command line
TEST_NAME = 'budget_test'   # Name of this test, for saving files
POOL = 12                   # Number of threads to spawn
TRAIN_SIZE = 0.8            # Ratio for the train set (including dev)
MIN_OCCURANCE = 0.05        # The minimum occurance to include in vocab
MAX_OCCURANCE = 1.0         # The minimum occurance to include in vocab
MAX_BUDGET = 20             # Max budget to try for each explainer
N_SAMPLES = 50              # Number of samples to evaluate explainer on

# Default log path
LOG_PATH = os.path.join('logs', TEST_NAME)

# Mapping of model names to model types
MODELS = {
        'mlp': MLPClassifier(),
        'logistic': LogisticRegression(solver='lbfgs'),
        'rf': RandomForestClassifier(n_estimators=50)
}

EXPLAINERS = {
        'Random': RandomExplainer,
        'Greedy': GreedyExplainer,
        'LIME': LimeExplainer,
        'SHAP(kmeans)': ShapExplainer,
        'SHAP(median)': ShapMedianExplainer,
        'SHAP(zeros)': ShapZerosExplainer,
}


def setup_argparse():
    desc = 'This script is meant to compare explainers ability recover bias ' \
            'that we have introduced to models.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
            'dataset',
            type=str,
            metavar='DATASET',
            help='CSV dataset to bias')
    parser.add_argument(
            'model',
            type=str,
            metavar='MODEL',
            help= ' | '.join(list(MODELS.keys())))
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
            '--log-dir',
            type=str,
            metavar='LOG_DIR',
            default=LOG_PATH,
            help='Log file directory (default = {})'.format(LOG_PATH))
    parser.add_argument(
            '--quiet',
            action='store_true',
            help='Do not print out information while running')
    return parser


def run_seed(seed):
    if args.quiet: sys.stdout = open(os.devnull, 'w')

    runlog = {}
    runlog['test_name'] = TEST_NAME

    # Setting seed #############################################################
    print('\n----------- Running SEED = {} -------------'.format(seed))
    np.random.seed(seed)
    runlog['seed'] = seed

    # Loading dataset ##########################################################
    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = utils.get_dataset(args.dataset, TRAIN_SIZE, runlog)

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

    train_df = pd.DataFrame(data=X_train, columns=feature_names)
    train_df['label'] = y_train

    test_df = pd.DataFrame(data=X_test, columns=feature_names)
    test_df['label'] = y_test

    # Resampling dataset #######################################################
    train_df, orig_balance = utils.resample(train_df, feature_names)

    # Randomly creating bias ###################################################
    bias_idx, bias_word, bias_class = utils.create_bias(
            train_df,
            test_df,
            feature_names,
            orig_balance,
            runlog
    )

    # Train a biased model to test on #########################################
    model_bias = utils.train_models(
            args.model,
            MODELS,
            train_df,
            runlog,
            bias_only=True
    )

    # Get data points to explain ###############################################
    eval_on_train = True
    if eval_on_train:
        mask = train_df.iloc[:, bias_idx] > 0
        R = train_df[mask]
        X_explain = train_df.drop(['label', 'label_bias'], axis=1).values
    else:
        mask = test_df.iloc[:, bias_idx] > 0
        R = test_df[mask]
        X_explain = test_df.drop(['label', 'label_bias'], axis=1).values

    n_samples = min(N_SAMPLES, len(X_explain))

    if args.model == 'logistic':
        EXPLAINERS['Ground Truth'] = LogisticExplainer

    for name in EXPLAINERS:
        runlog['explainer'] = name
        runlog['n_samples'] = name

        print('\tEXPLAINER = {}'.format(name))
        print('\t\tNUM_SAMPLES = {}'.format(n_samples))

        constructor = EXPLAINERS[name]
        explainer = constructor(
                model_bias,
                X_train,
                feature_names,
                seed
        )

        for budget in range(1, MAX_BUDGET + 1):
            runlog['budget'] = budget
            correct = 0
            incorrect = 0
            tp_error = 0
            fn_error = 0
            for i in range(n_samples):
                instance = X_explain[i]
                top_feats, error = explainer.explain(instance, budget)
                print('\t\tIMPORTANT_FEATURES =\n\t\t' + str(top_feats))
                if bias_word in top_feats:
                    tp_error += error
                    correct += 1
                else:
                    fn_error += error
                    incorrect += 1

            # Compute faithfulness w/ recall
            tp_error = tp_error / correct if correct > 0 else -1
            fn_error = fn_error / incorrect if incorrect > 0 else -1
            recall = correct / (correct + incorrect)

            runlog['tp_error'] = tp_error
            runlog['fn_error'] = fn_error
            runlog['recall'] = recall
            print('\t\tTP_ERROR = {:.4f}'.format(tp_error))
            print('\t\tFN_ERROR = {:.4f}'.format(fn_error))
            print('\t\tRECALL   = {:.4f}'.format(recall))

            filename = '{:s}_{:03d}_{:02d}.json'.format(name, seed, budget)
            utils.save_log(LOG_PATH, filename, runlog)


# Parse the command line arguments and start running seeds
if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    assert (args.model in MODELS), "Unrecognized model: " + args.model

    if POOL > 1:
        Pool(POOL).map(run_seed, range(args.seed_low, args.seed_high))
    else:
        for i in range(args.seed_low, args.seed_high):
            run_seed(i)
