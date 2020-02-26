import os
os.environ['MKL_NUM_THREADS'] = '1'

import linecache
import utils
import biases
from models import (
        WeightedNeuralNet,
        DNN,
        LSTM
)
from explainers import(
        LimeExplainer,
        ShapExplainer,
        GreedyExplainer,
        LogisticExplainer,
        TreeExplainer,
        RandomExplainer,
)

import sys
import time
import json
import pprint
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
torch.set_num_threads(1)
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


# GLOBALS ######################################################################

global args                     # Arguments from cmd line
LOG_PATH = 'logs'               # Top level directory for log files
POOL_SIZE = 10                  # How many workers to spawn (one per seed)
TRAIN_SIZE = 0.9                # Train split ratio (including dev)
MIN_OCCURANCE = 0.05            # Min occurance for words to be vectorized
MAX_OCCURANCE = 1.00            # Max occurance for words to be vectorized
BIAS_MIN_DF = 0.30              # Min occurance for words to be bias words
BIAS_MAX_DF = 0.50              # Max occurance for words to be bias words
MAX_BUDGET = 5                  # Upper bound of budget to test explainers
N_SAMPLES = 50                  # Number of samples to evaluate each exapliner

# Test types handled by this script
TESTS = [
        'budget_test', 
        'bias_test'
]

assert BIAS_MAX_DF < MAX_OCCURANCE, 'Model will not see bias word'
assert BIAS_MIN_DF > MIN_OCCURANCE, 'Model will not see bias word'

# DNN Features
DNN_MAX_VOCAB = 30
DNN_N_HIDDEN = 50
DNN_MAX_EPOCHS = 50
DNN_LR = 0.01

# # LSTM Features
# LSTM_MAX_VOCAB = 300
# LSTM_PAD_LEN = 300
# LSTM_N_HIDDEN = 300
# LSTM_N_EMBED = 300
# LSTM_MAX_EPOCHS = 100
# LSTM_LR = 0.01

# LSTM callbacks
# EPOCH_SCORE = callbacks.EpochScoring(
#         scoring='f1',
#         lower_is_better=False,
#         name='valid_f1')
# CHECKPOINT = callbacks.Checkpoint(
#         monitor='valid_f1_best',
#         dirname='saved_models')
# SCHEDULER = callbacks.LRScheduler(
#         policy='ReduceLROnPlateau',
#         monitor='valid_loss')
# STOPPER = callbacks.EarlyStopping(
#         monitor='valid_loss',
#         threshold=0.001)

# class TextFeaturizer:
#     def __init__(self, max_vocab):
#         self.max_vocab = max_vocab
#         self.tokenizer = Tokenizer(num_words=max_vocab)
#
#     def fit(self, X, y, **fit_params):
#         self.tokenizer.fit_on_texts(X)
#         return self
#
#     def transform(self, instances):
#         return self.tokenizer.texts_to_sequences(instances)
#
#     def inverse_transform(self, instances):
#         instances = np.array(instances)
#         rev_dict = dict(map(reversed, self.tokenizer.word_index.items()))
#         res = []
#         if len(instances.shape) == 2:
#             for instance in instances:
#                 inverse = [rev_dict[token] for token in instance]
#                 res.append(inverse)
#         else:
#             res = [rev_dict[token] for token in instance]
#         return res
#
#     def get_feature_names(self):
#         feats = list(self.tokenizer.word_index.keys())[:self.max_vocab]
#         print(feats)
#         return feats


# def r_accuracy(net, ds, y=None):
#     pred = net.predict(ds)
#     y_true = []
#     y_pred = []
#     for i, (x, y) in enumerate(ds):
#         if x['biased']:
#             y_pred.append(pred[i])
#             y_true.append(y)
#     return accuracy_score(y_true, y_pred)
#
# def nr_accuracy(net, ds, y=None):
#     pred = net.predict(ds)
#     y_true = []
#     y_pred = []
#     for i, (x, y) in enumerate(ds):
#         if not x['biased']:
#             y_pred.append(pred[i])
#             y_true.append(y)
#     return accuracy_score(y_true, y_pred)
#

# epoch_r_acc = callbacks.EpochScoring(r_accuracy, lower_is_better=False)
# epoch_nr_acc = callbacks.EpochScoring(nr_accuracy, lower_is_better=False)

# Mapping of model names to model objects
MODELS = {
    'logistic': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            max_features=DNN_MAX_VOCAB,
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
            max_features=DNN_MAX_VOCAB,
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
            max_features=DNN_MAX_VOCAB,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', DecisionTreeClassifier()),
    ]),

    'dnn': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            max_features=DNN_MAX_VOCAB,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', WeightedNeuralNet(
            module=DNN,
            device='cuda',
            module__n_input=DNN_MAX_VOCAB,
            max_epochs=DNN_MAX_EPOCHS,
            lr=DNN_LR,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='valid_loss',
                    threshold=0.001),
                callbacks.LRScheduler(
                    policy='ReduceLROnPlateau',
                    monitor='valid_loss')
            ]))
    ]),

    # 'lstm': lambda: Pipeline([
    #     ('text2ind', TextFeaturizer(LSTM_MAX_VOCAB)),
    #     ('padder', Padder2d(
    #         max_len=LSTM_PAD_LEN,
    #         pad_value=0,
    #         dtype=int)),
    #     ('model', WeightedNeuralNet(
    #         module=LSTM,
    #         device='cuda',
    #         callbacks=[
    #             epoch_r_acc,
    #             epoch_nr_acc,
    #             EPOCH_SCORE,
    #             STOPPER,
    #             SCHEDULER],
    #         module__n_input=LSTM_MAX_VOCAB+1,
    #         module__n_pad=LSTM_PAD_LEN,
    #         module__n_embedding=LSTM_N_EMBED,
    #         module__n_hidden=LSTM_N_HIDDEN,
    #         max_epochs=LSTM_MAX_EPOCHS,
    #         iterator_train__shuffle=True,
    #         lr=LSTM_LR))
    # ])
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
    runlog['seed']       = seed
    runlog['test_name']  = args.test_type
    runlog['model_type'] = args.model
    runlog['bias_len']   = args.bias_length
    runlog['min_occur']  = MIN_OCCURANCE
    runlog['max_occur']  = MIN_OCCURANCE

    EXPLAINERS = {
        # 'Random': RandomExplainer,
        # 'Greedy': GreedyExplainer,
        # 'LIME': LimeExplainer,
        # 'SHAP': ShapExplainer,
    }

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
            args.bias_length,
            BIAS_MIN_DF,
            BIAS_MAX_DF,
            None,
            runlog
    )

    labels_train_bias, \
    biased_train,      \
    flipped_train = bias_obj.bias(reviews_train, labels_train)

    labels_test_bias, \
    biased_test,      \
    flipped_test = bias_obj.bias(reviews_test, labels_test)

    # Convert to pandas df
    columns = ['reviews', 'label_orig', 'label_bias', 'biased', 'flipped']
    train_data = zip(
            reviews_train,
            labels_train,
            labels_train_bias,
            biased_train,
            flipped_train)
    test_data = zip(
            reviews_test,
            labels_test,
            labels_test_bias,
            biased_test,
            flipped_test)
    train_df = pd.DataFrame(data=train_data, columns=columns)
    test_df = pd.DataFrame(data=test_data, columns=columns)

    # Training biased model ####################################################
    model_type = MODELS[args.model]
    model_orig, model_bias = utils.train_models(model_type, train_df, runlog)

    # Standard evaluation of both models on test set ###########################
    utils.evaluate_models_test(model_orig, model_bias, test_df, runlog)

    # Evaluate both models on biased region R and ~R ###########################
    utils.evaluate_models(model_orig, model_bias, test_df, runlog)
    if (not args.no_log) and args.test_type == 'bias_test':
        filename = '{0}_{1:04d}.json'.format(runlog['bias_len'], runlog['seed'])
        utils.save_log(args.log_dir, filename, runlog)

    if args.test_type == 'bias_test':
        return

    # Get data points to test explainer on #####################################
    # explain = train_df[ train_df['biased'] ]    # region R
    flipped = train_df[ train_df['biased'] & train_df['flipped'] ]
    unflipped = train_df[ train_df['biased'] & ~train_df['flipped'] ]
    print('\t\tNUM_EXPLAIN = {}'.format(len(flipped) + len(unflipped)))

    X_explain_flipped = flipped['reviews'].values
    X_explain_unflipped = unflipped['reviews'].values

    # X_explain = explain['reviews'].values
    n_samples = min(N_SAMPLES, len(X_explain_flipped), len(X_explain_unflipped))
    runlog['n_samples'] = n_samples
    print('\t\tNUM_SAMPLES = {}'.format(n_samples))

    # Handle interpretable models by adding their respective explainer
    if args.model == 'logistic':
        EXPLAINERS['Ground Truth'] = LogisticExplainer
    elif args.model == 'dt':
        EXPLAINERS['Ground Truth'] = TreeExplainer
    for name in EXPLAINERS:
        runlog['explainer'] = name
        print('\tEXPLAINER = {}'.format(name))

        explainer = EXPLAINERS[name](model_bias, reviews_train, seed)

        for budget in range(1, MAX_BUDGET + 1):
            runlog['budget'] = budget
            recall_sum = 0
            flipped_recall_sum = 0
            unflipped_recall_sum = 0
            for i in range(n_samples):
                importance_pairs = explainer.explain(X_explain_flipped[i], budget)
                runlog['top_features'] = [str(feat) for feat, _ in importance_pairs]
                runlog['feature_importances'] = [float(imp) for _, imp in importance_pairs]
                top_feats = [feat for (feat, importance) in importance_pairs]
                recall = 0
                for word in bias_obj.bias_words:
                    if word in top_feats:
                        recall += 1
                recall /= args.bias_length
                recall_sum += recall
                flipped_recall_sum += recall

            for i in range(n_samples):
                importance_pairs = explainer.explain(X_explain_unflipped[i], budget)
                runlog['top_features'] = [str(feat) for feat, _ in importance_pairs]
                runlog['feature_importances'] = [float(imp) for _, imp in importance_pairs]
                top_feats = [feat for (feat, importance) in importance_pairs]
                recall = 0
                for word in bias_obj.bias_words:
                    if word in top_feats:
                        recall += 1
                recall /= args.bias_length
                recall_sum += recall
                unflipped_recall_sum += recall

            avg_recall = recall_sum / (2 * n_samples)
            runlog['recall'] = avg_recall
            print('\t\tAVG_RECALL   = {:.4f}'.format(avg_recall))

            flipped_avg_recall = flipped_recall_sum / n_samples
            runlog['flipped_recall'] = flipped_avg_recall
            print('\t\tFLIP_AVG_RECALL   = {:.4f}'.format(flipped_avg_recall))

            unflipped_avg_recall = unflipped_recall_sum / n_samples
            runlog['unflipped_recall'] = unflipped_avg_recall
            print('\t\tUNFLIP_AVG_RECALL = {:.4f}'.format(unflipped_avg_recall))

            if (not args.no_log) and args.test_type == 'budget_test':
                filename = '{:s}_{:d}_{:03d}_{:02d}.json'.format(
                        name, args.bias_length, seed, budget)
                utils.save_log(LOG_PATH, filename, runlog)



def setup_argparse():
    desc = 'This script compares multiple explainers\' ability to' \
           'recover bias that we have trained into models.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
            'test_type',
            type=str,
            metavar='TEST',
            help=' | '.join(TESTS))
    parser.add_argument(
            'dataset',
            type=str,
            metavar='DATASET',
            help='CSV dataset to bias')
    parser.add_argument(
            'model',
            type=str,
            metavar='MODEL',
            help=' | '.join(list(MODELS.keys())))
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
            metavar='BIAS_LENGTH',
            help='Number of features to include in bias')
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
    if POOL_SIZE > 1 and len(seeds) > 1 and (not args.single_thread):
        pool = Pool(POOL_SIZE)
        pool.map(run_seed, seeds)
        pool.close()
        pool.join()
    else:
        for seed in seeds:
            torch.cuda.empty_cache()
            run_seed(seed)

            # objs = []
            # for obj in gc.get_objects():
            #     try:
            #         mem = sys.getsizeof(obj)
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             objs.append((mem, '{} {}'.format(type(obj),
            #                 obj.size())))
            #         elif isinstance(obj, dict):
            #             objs.append((mem, obj.keys()[0]))
            #         elif isinstance(obj, list):
            #             if len(obj) > 5:
            #                 objs.append((mem, obj[:5]))
            #             else:
            #                 objs.append((mem, obj))
            #         else:
            #             objs.append((mem, '{}'.format(type(obj))))
            #     except:
            #         pass
            #
            # sort_objs = sorted(objs, key=lambda x: x[0], reverse=True)
            # print('Top mem users:')
            # for o in sort_objs[:100]:
            #     print('\t' + str(o))
