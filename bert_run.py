import os
import sys
import json
import shutil
import argparse
from multiprocessing import Pool
from allennlp.training.trainer import Trainer
from allennlp.commands.train import train_model_from_file
from sklearn.model_selection import train_test_split

import tqdm
import numpy as np
import pandas as pd
import torch

import utils
import biases
from models import pipelines
from bertmodel import RobertaLarge
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
TRAIN_SIZE = 0.6                # Train split ratio
VALID_SIZE = 0.2                # Validation split ratio
TEST_SIZE  = 0.2                # Test split ratio
# MIN_OCCURANCE = 0.05            # Min occurance for words to be vectorized
# MAX_OCCURANCE = 1.00            # Max occurance for words to be vectorized
BIAS_MIN_DF = 0.20              # Min occurance for words to be bias words
BIAS_MAX_DF = 0.60              # Max occurance for words to be bias words
MAX_BUDGET = 5                  # Upper bound of budget to test explainers
N_SAMPLES = 50                  # Number of samples to evaluate each explainer
N_BAGS = 3                      # Number of bags to create in bagging test
MIN_R_PERFOMANCE = 0.90         # Minimum accuracy on region R to allow
MIN_F1_SCORE = 0.50             # Minimum F1-score to allow for biased model
MAX_RETRIES = 1                 # Maximum retries if model performance is low
BIAS_LENS = range(1, 2)         # Range of bias lengths to run


# BERT specific changes
MODEL_TYPE = 'roberta'
TMP_DIR = os.path.join('/tmp', 'bert_data')

 # Path to toy dataset for testing this scripts functionality
TOY_DATASET = 'datasets/imdb.csv'

# Test types handled by this script
TESTS = [ 'bias_test', 'budget_test', 'boost_test' ]


def main():
    if args.quiet: sys.stdout = open(os.devnull, 'w')
    pool_size = args.n_workers
    seeds = range(args.seed_low, args.seed_high)

    datasets = []
    if args.toy:    # only test on 1 small dataset
        datasets.append(TOY_DATASET)
    else:
        for filename in os.listdir(args.data_dir):
            datasets.append(os.path.join(args.data_dir, filename))

    # Build list of arguments
    arguments = []
    for seed in range(args.seed_low, args.seed_high):
        for dataset in datasets:
            for bias_len in BIAS_LENS:
                arguments.append({
                    'seed': seed,
                    'dataset': dataset,
                    'model_type': MODEL_TYPE,
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
    model_type = 'roberta'
    bias_length = arguments['bias_length']
    dataset_name = dataset.split('/')[-1].split('.csv')[0]

    if 'train_attempts' not in arguments:
        np.random.seed(seed)
        os.environ['MKL_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        arguments['train_attempts'] = 1

    elif arguments['train_attempts'] > MAX_RETRIES:
        # assert False, 'Exceeded maximum number of retries, bias failed'
        pass

    print('\tTRAIN_ATTEMPTS = {}'.format(arguments['train_attempts']))

    # Building Runlog dictionary with seed args
    runlog = {}
    runlog['toy']        = args.toy
    runlog['seed']       = seed
    runlog['test_name']  = args.test
    runlog['model_type'] = model_type
    runlog['dataset']    = dataset_name
    runlog['bias_len']   = bias_length
    # runlog['min_occur']  = MIN_OCCURANCE
    # runlog['max_occur']  = MAX_OCCURANCE

    SERIAL_DIR = 'save'
    ORIG_NAME = 'orig_model'
    BIAS_NAME = 'bias_model'
    PARAM_FILE = 'roberta.jsonnet'

    orig_train_path, \
    orig_valid_path, \
    orig_test_path,  \
    bias_train_path, \
    bias_valid_path, \
    bias_test_path,  \
    train_df,        \
    valid_df,        \
    test_df,         \
    bias_words = split_dataset(dataset, bias_length, runlog)

    orig_model_path = os.path.join(SERIAL_DIR, ORIG_NAME)
    bias_model_path = os.path.join(SERIAL_DIR, BIAS_NAME)

    if not args.recover:
        if os.path.exists(orig_model_path):
            shutil.rmtree(orig_model_path)
        if os.path.exists(bias_model_path):
            shutil.rmtree(bias_model_path)

    orig_overrides = json.dumps({
        'train_data_path'      : orig_train_path,
        'validation_data_path' : orig_valid_path,
        'test_data_path'       : orig_test_path,
    })

    train_model_from_file(
        parameter_filename=PARAM_FILE,
        serialization_dir=orig_model_path,
        overrides=orig_overrides,
        recover=args.recover,
    )

    bias_overrides = json.dumps({
        'train_data_path'      : bias_train_path,
        'validation_data_path' : bias_valid_path,
        'test_data_path'       : bias_test_path,
    })

    train_model_from_file(
        parameter_filename=PARAM_FILE,
        serialization_dir=bias_model_path,
        overrides=bias_overrides,
        recover=args.recover,
    )


    orig_model = RobertaLarge(model_path=orig_model_path)
    bias_model = RobertaLarge(model_path=bias_model_path)

    # Evaluate both models on biased region R and ~R
    utils.evaluate_models(orig_model, bias_model, test_df, runlog, quiet=args.quiet)

    if args.test == 'bias_test':
        if not args.no_log:
            utils.save_log(args.log_dir, runlog, quiet=args.quiet)
        return

    # TODO re-add checks

    # R_bias_acc = runlog['results'][1][0]
    # runlog['bias_test_lfr'] = R_bias_acc
    # bias_f1 = runlog['bias_test_f1']

    # if R_bias_acc < MIN_R_PERFOMANCE and arguments['train_attempts'] <= MAX_RETRIES:
    #     print('Accuracy on region R too low (expected >= {}, got {})'.format(
    #         MIN_R_PERFOMANCE, R_bias_acc))
    #     arguments['train_attempts'] += 1
    #     run_seed(arguments)
    #     return
    #
    # if bias_f1 < MIN_F1_SCORE and arguments['train_attempts'] <= MAX_RETRIES:
    #     print('F1-score too low on biased model (expected >= {}, got {})'.format(
    #         MIN_F1_SCORE, bias_f1))
    #     arguments['train_attempts'] += 1
    #     run_seed(arguments)
    #     return

    if args.test == 'budget_test':
        exps = {
            # 'Random': RandomExplainer,
            'Greedy': GreedyExplainer,
            'LIME': LimeExplainer,
            'SHAP':ShapExplainer,
        }
        n_samples = 1 if args.toy else N_SAMPLES
        explainers_budget_test(bias_model, exps, bias_words, train_df, n_samples, runlog)
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


def split_dataset(dataset_path, bias_length, runlog, quiet=False):
    # Load and split full dataset
    with open(dataset_path, 'r') as f:
        data = pd.read_csv(dataset_path, header=None, names=['reviews', 'labels'])

    train_val_data, test_data = train_test_split(data, test_size=0.2)
    train_data, val_data = train_test_split(train_val_data, train_size=0.5)

    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    reviews_train = train_data['reviews'].values
    labels_train= train_data['labels'].values

    bias_obj = biases.ComplexBias(
            reviews_train,
            labels_train,
            bias_length,
            BIAS_MIN_DF,
            BIAS_MAX_DF,
            runlog,
            quiet=quiet)

    # Build dataframes with orig and bias labels
    train_df = bias_obj.build_df_from_df(train_data, runlog)
    valid_df = bias_obj.build_df_from_df(val_data, runlog)
    test_df  = bias_obj.build_df_from_df(test_data, runlog)

    train_df = utils.oversample(train_df)
    valid_df = utils.oversample(valid_df)
    test_df = utils.oversample(test_df)

    FAST_FRAC = 0.2
    train_df = train_df.sample(frac=FAST_FRAC, replace=False)
    valid_df = valid_df.sample(frac=0.01, replace=False)
    test_df = test_df.sample(frac=FAST_FRAC, replace=False)


    orig_train_path = os.path.join(TMP_DIR, 'orig_train.csv')
    orig_valid_path = os.path.join(TMP_DIR, 'orig_valid.csv')
    orig_test_path  = os.path.join(TMP_DIR, 'orig_test.csv')

    bias_train_path = os.path.join(TMP_DIR, 'bias_train.csv')
    bias_valid_path = os.path.join(TMP_DIR, 'bias_valid.csv')
    bias_test_path  = os.path.join(TMP_DIR, 'bias_test.csv')

    orig_cols = ['reviews', 'label_orig']
    bias_cols = ['reviews', 'label_bias']

    train_df.to_csv(orig_train_path, header=False, index=False, columns=orig_cols)
    valid_df.to_csv(orig_valid_path, header=False, index=False, columns=orig_cols)
    test_df.to_csv(orig_test_path, header=False, index=False, columns=orig_cols)

    train_df.to_csv(bias_train_path, header=False, index=False, columns=bias_cols)
    valid_df.to_csv(bias_valid_path, header=False, index=False, columns=bias_cols)
    test_df.to_csv(bias_test_path, header=False, index=False, columns=bias_cols)

    return orig_train_path, \
           orig_valid_path, \
           orig_test_path,  \
           bias_train_path, \
           bias_valid_path, \
           bias_test_path,  \
           train_df,        \
           valid_df,        \
           test_df,         \
           bias_obj.bias_words


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

    #TODO test and remove
    print('Form of input:')
    print(X_explain[0])

    print('integrate:')
    res = model_bias.explain(X_explain[0], method='integrate')
    print(res)

    print('simple:')
    res = model_bias.explain(X_explain[0], method='simple')
    print(res)
    exit()

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
        '--data-dir',
        type=str,
        metavar='DATA_DIR',
        default=DATA_DIR,
        help='Dataset directory (default = {})'.format(DATA_DIR))
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
    parser.add_argument(
        '--recover',
        action='store_true',
        help='Try to recover from a trained model')

    args = parser.parse_args()

    bad_test_msg = 'Test not found: {}'.format(args.test)
    assert (args.test in TESTS)

    bad_seed_msg = 'No seeds in [{}, {})'.format(args.seed_low, args.seed_high)
    assert (args.seed_low < args.seed_high), bad_seed_msg

    return args


if __name__ == '__main__':
    args = setup_args()
    main()
