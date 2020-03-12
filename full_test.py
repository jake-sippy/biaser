import os
import utils
import sys
import time
import json
import tqdm
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

import sklearn
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        recall_score,
        f1_score
)

from skorch import callbacks

import biases
from models import WeightedNeuralNet, MLP, LSTM
from explainers import(
        LimeExplainer,
        ShapExplainer,
        GreedyExplainer,
        LogisticExplainer,
        TreeExplainer,
        RandomExplainer,
)

# GLOBALS ######################################################################

global args                     # Arguments from cmd line
LOG_PATH = 'logs'               # Top level directory for log files
POOL_SIZE = 5                   # How many workers to spawn (one per seed)
TRAIN_SIZE = 0.9                # Train split ratio (including dev)
MIN_OCCURANCE = 0.05            # Min occurance for words to be vectorized
MAX_OCCURANCE = 1.00            # Max occurance for words to be vectorized
BIAS_MIN_DF = 0.30              # Min occurance for words to be bias words
BIAS_MAX_DF = 0.50              # Max occurance for words to be bias words
MAX_BUDGET = 5                  # Upper bound of budget to test explainers
N_SAMPLES = 50                  # Number of samples to evaluate each explainer

# Test types handled by this script
TESTS = [ 'budget_test', 'bias_test' ]

assert BIAS_MAX_DF < MAX_OCCURANCE, 'Model will not see bias word'
assert BIAS_MIN_DF > MIN_OCCURANCE, 'Model will not see bias word'

# MLP Features learned through CV
MLP_MAX_VOCAB = 150
MLP_N_HIDDEN = 50
MLP_MAX_EPOCHS = 50
MLP_LR = 0.01

# Model names to constructors
MODELS = {
    'logistic': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', LogisticRegression(solver='lbfgs')),
    ]),

    'rf': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', RandomForestClassifier(n_estimators=50)),
    ]),

    'dt': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', DecisionTreeClassifier()),
    ]),

    'mlp': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            max_features=MLP_MAX_VOCAB,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', WeightedNeuralNet(
            module=MLP,
            device='cuda',
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='valid_loss',
                    threshold=0.001),
                callbacks.LRScheduler(
                    policy='ReduceLROnPlateau',
                    monitor='valid_loss')
            ],
            module__n_input=MLP_MAX_VOCAB,
            max_epochs=MLP_MAX_EPOCHS,
            lr=MLP_LR))
    ]),
}





def run_seed(arguments):
    seed = arguments['seed']
    dataset = arguments['dataset']
    model_type = arguments['model_type']
    bias_length = arguments['bias_length']
    
    explainers = {
        'Random': RandomExplainer,
        'Greedy': GreedyExplainer,
        'LIME': LimeExplainer,
        'SHAP': ShapExplainer,
    }

    runlog = {}
    runlog['seed']       = seed
    runlog['test_name']  = args.test_type
    runlog['model_type'] = model_type
    runlog['bias_len']   = bias_length
    runlog['min_occur']  = MIN_OCCURANCE
    runlog['max_occur']  = MIN_OCCURANCE

    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    np.random.seed(seed)

    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = utils.load_dataset(dataset, TRAIN_SIZE, runlog,
            quiet=True)

    # Create bias #############################################################
    bias_obj = biases.ComplexBias(
            reviews_train,
            labels_train,
            bias_length,
            BIAS_MIN_DF,
            BIAS_MAX_DF,
            runlog,
            quiet=True
    )

    train_df = bias_obj.build_df(reviews_train, labels_train)
    test_df = bias_obj.build_df(reviews_test, labels_test)

    # Training biased model ####################################################
    model = MODELS[model_type]
    model_orig, model_bias = utils.train_models(model, train_df,
            runlog, quiet=True)

    # Standard evaluation of both models on test set ###########################
    utils.evaluate_models_test(model_orig, model_bias, test_df, runlog,
            quiet=True)

    # Evaluate both models on biased region R and ~R ###########################
    utils.evaluate_models(model_orig, model_bias, test_df, runlog, quiet=True)
    if (not args.no_log) and args.test_type == 'bias_test':
        filename = '{0}_{1:04d}.json'.format(runlog['bias_len'], runlog['seed'])
        utils.save_log(args.log_dir, filename, runlog, quiet=True)

    if args.test_type == 'bias_test': return

    # Get data points to test explainer on #####################################
    explain = train_df[ train_df['biased'] & train_df['flipped'] ]
    X_explain = explain['reviews'].values
    n_samples = min(N_SAMPLES, len(explain))
    runlog['n_samples'] = n_samples

    # Handle interpretable models by adding their respective explainer #########
    if model_type == 'logistic':
        explainers['Ground Truth'] = LogisticExplainer
    elif model_type == 'dt':
        explainers['Ground Truth'] = TreeExplainer

    # Test recall of explainers ################################################
    for name in explainers:
        runlog['explainer'] = name
        explainer = explainers[name](model_bias, reviews_train, seed)
        for budget in range(1, MAX_BUDGET + 1):
            runlog['budget'] = budget
            avg_recall = 0
            for i in range(n_samples):
                importance_pairs = explainer.explain(X_explain[i], budget)
                top_feats = [str(feat) for feat, _ in importance_pairs]
                importances = [float(imp) for _, imp in importance_pairs]
                runlog['top_features'] = top_feats
                runlog['feature_importances'] = importances
                recall = 0
                for word in bias_obj.bias_words:
                    if word in runlog['top_features']:
                        recall += 1
                avg_recall += recall / bias_length

            avg_recall /= n_samples
            runlog['recall'] = avg_recall

            if (not args.no_log) and args.test_type == 'budget_test':
                filename = '{:s}_{:d}_{:03d}_{:02d}.json'.format(
                        name, bias_length, seed, budget)
                utils.save_log(LOG_PATH, filename, runlog, quiet=True)


def setup_args():
    desc = 'This script compares multiple explainers\' ability to' \
           'recover bias that we have trained into models.'
    parser = argparse.ArgumentParser(description=desc)
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
            'n_workers',
            type=int,
            metavar='N_WORKERS',
            help='Number of workers to spawn')
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
    parser.add_argument(
            '--no-log',
            action='store_true',
            help='Do not log information while running')
    parser.add_argument(
            '--single-thread',
            action='store_true',
            help='Force single-thread for multiple seeds')

    args = parser.parse_args()
    bad_seed_msg = 'No seeds in [{}, {})'.format(args.seed_low, args.seed_high)
    assert (args.seed_low < args.seed_high), bad_seed_msg
    return args


if __name__ == '__main__':
    args = setup_args()
    if args.quiet: sys.stdout = open(os.devnull, 'w')

    pool_size = args.n_workers

    seeds = range(args.seed_low, args.seed_high)
    args.test_type = 'budget_test'

    arguments = []
    DATA_DIR = 'datasets'
    for f in os.listdir(DATA_DIR):
        dataset = os.path.join(DATA_DIR, f)
        for model_type in ['logistic', 'dt', 'rf']:
            for seed in range(args.seed_low, args.seed_high):
                arguments.append({
                    'seed': seed,
                    'dataset': dataset,
                    'model_type': model_type,
                    'bias_length': 1
                })

                arguments.append({
                    'seed': seed,
                    'dataset': dataset,
                    'model_type': model_type,
                    'bias_length': 2
                })
    pool = Pool(pool_size, maxtasksperchild=1)
    list(tqdm.tqdm(pool.imap(run_seed, arguments, chunksize=1),
        total=len(arguments)))
    pool.close()
    pool.join()
