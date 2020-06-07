import os
import sys
import argparse
from multiprocessing import Pool

import tqdm
import numpy as np
import pandas as pd
import torch

import utils
import biases
from models import pipelines
from explainers import (
    GreedyExplainer,
    LimeExplainer,
    BaggedLimeExplainer,
    BaggedShapExplainer,
    ShapExplainer,
    RandomExplainer,
    TreeExplainer,
    LogisticExplainer
)

# GLOBALS ######################################################################
global args                     # Arguments from cmd line
LOG_PATH = 'logs'               # Top level directory for log files
DATA_DIR = 'datasets'           # Folder containing datasets
TRAIN_SIZE = 0.8                # Train split ratio (including dev)
MIN_OCCURANCE = 0.05            # Min occurance for words to be vectorized
MAX_OCCURANCE = 1.00            # Max occurance for words to be vectorized
BIAS_MIN_DF = 0.20              # Min occurance for words to be bias words
BIAS_MAX_DF = 0.80              # Max occurance for words to be bias words
MAX_BUDGET = 5                  # Upper bound of budget to test explainers
N_SAMPLES = 50                  # Number of samples to evaluate each explainer
N_BAGS = 3                      # Number of bags to create in bagging test
MIN_R_PERFOMANCE = 0.90         # Minimum accuracy on region R to allow
MIN_F1_SCORE = 0.50             # Minimum F1-score to allow for biased model
MAX_RETRIES = 5                 # Maximum retries if model performance is low
BIAS_LENS = range(1, 3)         # Range of bias lengths to run

 # Path to toy dataset for testing this scripts functionality
TOY_DATASET = 'datasets/newsgroups_atheism.csv'
# TOY_DATASET = 'datasets/imdb.csv'
# TOY_DATASET = 'datasets/goodreads.csv'

MODELS = [
    'logistic',
    'dt',
    'rf',
    'xgb'
    # 'mlp',
]

# Test types handled by this script
TESTS = [ 'bias_test', 'budget_test', 'boost_test' ]

assert BIAS_MAX_DF < MAX_OCCURANCE, 'Model will not see bias word'
assert BIAS_MIN_DF > MIN_OCCURANCE, 'Model will not see bias word'


def main():
    if args.quiet: sys.stdout = open(os.devnull, 'w')
    pool_size = args.n_workers
    seeds = range(args.seed_low, args.seed_high)

    datasets = []
    if args.toy:
        # Toy flag, only test on 1 small dataset
        datasets.append(TOY_DATASET)
    else:
        for filename in os.listdir(DATA_DIR):
            datasets.append(os.path.join(DATA_DIR, filename))

    # Build list of arguments
    arguments = []
    for seed in range(args.seed_low, args.seed_high):
        for dataset in datasets:
            for model_type in MODELS:
                for bias_len in BIAS_LENS:
                    arguments.append({
                        'seed': seed,
                        'dataset': dataset,
                        'model_type': model_type,
                        'bias_length': bias_len
                    })

    if pool_size == 1:
        for arg in arguments:
            run_seed(arg)
    else:
        pool = Pool(pool_size, maxtasksperchild=1)
        imap_results = pool.imap(run_seed, arguments, chunksize=1)
        list(tqdm.tqdm(imap_results, total=len(arguments)))
        pool.close()
        pool.join()


def run_seed(arguments):
    seed = arguments['seed']
    dataset = arguments['dataset']
    model_type = arguments['model_type']
    bias_length = arguments['bias_length']

    if 'train_attempts' not in arguments:
        np.random.seed(seed)
        os.environ['MKL_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        arguments['train_attempts'] = 1

    elif arguments['train_attempts'] > MAX_RETRIES:
        pass
        # assert False, 'Exceeded maximum number of retries, bias failed'

    print('\tTRAIN_ATTEMPTS = {}'.format(arguments['train_attempts']))

    # Building Runlog dictionary with seed args
    runlog = {}
    runlog['toy']        = args.toy
    runlog['seed']       = seed
    runlog['test_name']  = args.test
    runlog['model_type'] = model_type
    runlog['bias_len']   = bias_length
    runlog['min_occur']  = MIN_OCCURANCE
    runlog['max_occur']  = MAX_OCCURANCE

    model_orig, \
    model_bias, \
    train_df,   \
    test_df,    \
    bias_words = build_biased_model(dataset, model_type, bias_length, runlog)

    # Evaluate both models on biased region R and ~R
    utils.evaluate_models(model_orig, model_bias, test_df, runlog, quiet=args.quiet)
    utils.evaluate_models_test(model_orig, model_bias, test_df, runlog, quiet=args.quiet)

    R_bias_acc = runlog['results'][1][0]
    bias_f1 = runlog['bias_test_f1']
    if R_bias_acc < MIN_R_PERFOMANCE:
        print('Accuracy on region R too low (expected >= {}, got {})'.format(
            MIN_R_PERFOMANCE, R_bias_acc))
        arguments['train_attempts'] += 1
        run_seed(arguments)
        return

    if bias_f1 < MIN_F1_SCORE:
        print('F1-score too low on biased model (expected >= {}, got {})'.format(
            MIN_F1_SCORE, bias_f1))
        arguments['train_attempts'] += 1
        run_seed(arguments)
        return

    if (not args.no_log) and args.test == 'bias_test':
        utils.save_log(args.log_dir, runlog, quiet=args.quiet)
        return

    if args.test == 'budget_test':
        exps = {
            # 'Random': RandomExplainer,
            'Greedy': GreedyExplainer,
            'LIME': LimeExplainer,
            'SHAP':ShapExplainer,
        }
        n_samples = 1 if args.toy else N_SAMPLES
        explainers_budget_test(model_bias, exps, bias_words, train_df, n_samples, runlog)
        return

    if args.test ==  'boost_test':
        exps = {
            'Greedy': GreedyExplainer,
            'LIME': LimeExplainer,
            'SHAP':ShapExplainer,
            'Aggregate (LIME x 3)': BaggedLimeExplainer,
            'Aggregate (SHAP x 3)': BaggedShapExplainer,
        }
        n_samples = 1 if args.toy else N_SAMPLES
        explainers_budget_test(model_bias, exps, bias_words, train_df, n_samples, runlog)
        return


def build_biased_model(dataset_path, model_type, bias_length, runlog):
    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = utils.load_dataset(dataset_path, TRAIN_SIZE, runlog, quiet=args.quiet)

    bias_obj = biases.ComplexBias(
            reviews_train,
            labels_train,
            bias_length,
            BIAS_MIN_DF,
            BIAS_MAX_DF,
            runlog,
            quiet=args.quiet)

    train_df = bias_obj.build_df(reviews_train, labels_train, runlog)
    test_df = bias_obj.build_df(reviews_test, labels_test, runlog)

    model_pipeline = pipelines[model_type]
    model_orig, model_bias = utils.train_models(model_pipeline, train_df, runlog,
            quiet=args.quiet)

    return model_orig, model_bias, train_df, test_df, bias_obj.bias_words


def explainers_budget_test(
    model_bias,
    explainers,
    bias_words,
    train_df,
    n_samples,
    runlog
):
    X_all = train_df['reviews'].values
    explain = train_df[ train_df['biased'] & train_df['flipped'] ]
    X_explain = explain['reviews'].values
    n_samples = min(n_samples, len(explain))
    runlog['n_samples'] = n_samples
    bias_length = len(bias_words)

    # Handle interpretable models by adding their respective explainer
    if runlog['model_type'] == 'logistic':
        explainers['Ground Truth'] = LogisticExplainer
    elif runlog['model_type'] == 'dt':
        explainers['Ground Truth'] = TreeExplainer

    # Compute recall of exapliners
    for explainer_name in explainers:
        runlog['explainer'] = explainer_name
        explainer = explainers[explainer_name](model_bias, X_all)
        for budget in range(1, MAX_BUDGET + 1):
            runlog['budget'] = budget

            # Compute the average recall over `n_samples` instances
            avg_recall = 0
            for i in range(n_samples):
                importance_pairs = explainer.explain(X_explain[i], budget)
                top_feats = [str(feat) for feat, _ in importance_pairs]
                importances = [float(imp) for _, imp in importance_pairs]
                runlog['top_features'] = top_feats
                runlog['feature_importances'] = importances
                runlog['example_id'] = i
                recall = 0
                for word in bias_words:
                    if word in top_feats:
                        recall += 1
                runlog['recall'] = recall / bias_length

                if not args.no_log:
                    utils.save_log(args.log_dir, runlog, quiet=args.quiet)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test',
        type=str,
        metavar='TEST',
        help=' | '.join(TESTS))
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
    parser.add_argument(
        '--toy',
        action='store_true',
        help='Run a toy version of the test')

    args = parser.parse_args()

    bad_test_msg = 'Test not found: {}'.format(args.test)
    assert (args.test in TESTS)

    bad_seed_msg = 'No seeds in [{}, {})'.format(args.seed_low, args.seed_high)
    assert (args.seed_low < args.seed_high), bad_seed_msg

    return args


if __name__ == '__main__':
    args = setup_args()
    main()
