# This module is to test if biasing the dataset in a simple way has
# the effect on model performances that we would hope to see.
#
# Reviews at this stage are passed in as lines of json,
# each line is one review of the form:
# {"text": ..., "label": ...}
#
# JUST A HEADS UP: This code is pretty messy once you get to main, it still
# needs to be broken up into readable methods.

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

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        recall_score,
        f1_score
)
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from interpret import show, preserve
from interpret.blackbox import (
        PartialDependence,
        ShapKernel,
        LimeTabular,
        MorrisSensitivity
)

# import shap
# from lime.lime_text import LimeTextExplainer

# The minimum occurance of words to include as proportion of reviews
MIN_OCCURANCE = 0.1

# The maximum occurance of words to include as proportion of reviews
MAX_OCCURANCE = 1.0

# Ratio to split for the train set (including dev)
TRAIN_SIZE = 0.8

# Should these runs be outputted in log files
LOGGING_ENABLED = True

# How many examples from the test set should be used to evaluate the explainer
N_EXPLAIN = 100

def setup_argparse():
    parser = argparse.ArgumentParser(
            description='This script is meant to show that we can reliably '  \
                        'introduce bias in a dataset, and the model that we ' \
                        'train on this dataset (jk its for lime).')
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
            default='budget_logs',
            help='Directory to save JSON log files ' \
                 '(default = budget_logs/)')
    parser.add_argument(
            '--verbose',
            action='store_true',
            help='Print out information while running')
    return parser


def oversample(df):
    counts = df['label_orig'].value_counts()
    smaller_class = df[ df['label_orig'] == counts.idxmin() ]
    larger_class = df[ df['label_orig'] == counts.idxmax() ]
    over = smaller_class.sample(counts.max(), replace=True)
    return pd.concat([over, larger_class], axis=0)


def evaluate_models(model_orig, model_bias, r, not_r):
    X_r = r.drop(['label_orig', 'label_bias'], axis=1).values
    y_r = r['label_orig'].values

    X_not_r = not_r.drop(['label_orig', 'label_bias'], axis=1).values
    y_not_r = not_r['label_orig'].values

    pred_orig_r = model_orig.predict(X_r)
    pred_orig_not_r = model_orig.predict(X_not_r)

    pred_bias_r = model_bias.predict(X_r)
    pred_bias_not_r = model_bias.predict(X_not_r)

    orig_r = accuracy_score(y_r, pred_orig_r)
    orig_not_r = accuracy_score(y_not_r, pred_orig_not_r)

    bias_r = accuracy_score(y_r, pred_bias_r)
    bias_not_r = accuracy_score(y_not_r, pred_bias_not_r)

    print('\tBIAS_RESULT = ')
    print('\t              R      !R')
    print('\torig model | {0:3.2f} | {1:3.2f}'.format(orig_r, orig_not_r))
    print('\tbias model | {0:3.2f} | {1:3.2f}'.format(bias_r, bias_not_r))
    return [[orig_r, orig_not_r], [bias_r, bias_not_r]]



def run_seed(seed):
    runlog = {}
    # Getting dataset name for log #############################################
    dataset_name = args.dataset.split('/')[-1].split('.csv')[0]
    runlog['dataset'] = dataset_name

    # Load dataset #############################################################
    data = pd.read_csv(args.dataset, header=None, names=['reviews', 'labels'])
    reviews = data['reviews'].astype(str).values
    labels = data['labels'].astype(int).values

    # Setting seed #############################################################
    # TODO: reintroduce looping over seeds
    # seed = args.seed_low
    print('\n----------- Running SEED = {} -------------'.format(seed))
    np.random.seed(seed)
    runlog['seed'] = seed

    # Splitting dataset ########################################################
    print('\tTRAIN_SIZE = {}'.format(TRAIN_SIZE))
    runlog['train_size'] = TRAIN_SIZE

    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = train_test_split(reviews, labels, train_size=TRAIN_SIZE)

    # Vectorizing and scaling dataset ##########################################
    print('\tMIN_OCCURANCE = {}'.format(MIN_OCCURANCE))
    print('\tMAX_OCCURANCE = {}'.format(MAX_OCCURANCE))
    runlog['min_occur'] = MIN_OCCURANCE
    runlog['max_occur'] = MAX_OCCURANCE

    # NOTE: with_mean=Fasle is required for scaling because the vector
    # representation is actually a sparse matrix, so mean cannot be calculated
    # it might be worth the extra memory to convert the vectorized rep. to a
    # dense matrix

    vectorizer = CountVectorizer(min_df=MIN_OCCURANCE, max_df=MAX_OCCURANCE)
    # vectorizer = TfidfVectorizer(min_df=MIN_OCCURANCE, max_df=MAX_OCCURANCE)
    scaler = StandardScaler(with_mean=False)
    preprocessor = make_pipeline(vectorizer, scaler)
    # preprocessor = make_pipeline(vectorizer)

    X_train = preprocessor.fit_transform(reviews_train).toarray()
    feature_names = vectorizer.get_feature_names()
    y_train = np.array(labels_train)
    train_df = pd.DataFrame(data=X_train, columns=feature_names)
    train_df['label_orig'] = y_train

    X_test = preprocessor.transform(reviews_test).toarray()
    y_test = np.array(labels_test)
    test_df = pd.DataFrame(data=X_test, columns=feature_names)
    test_df['label_orig'] = y_test

    print('\tFEATURES EXTRACTED = {}'.format(len(feature_names)))

    # Resample to balance training data ########################################

    orig_value_counts = train_df['label_orig'].value_counts()
    train_df = oversample(train_df)
    bias_value_counts = train_df['label_orig'].value_counts()

    print('\tORIGINAL BALANCE = ')
    print('\t\tClass_0 = {}\n\t\tClass_1 = {}'
            .format(orig_value_counts[0], orig_value_counts[1]))
    print('\tCORRECTED BALANCE = ')
    print('\t\tClass_0 = {}\n\t\tClass_1 = {}'
            .format(bias_value_counts[0], bias_value_counts[1]))

    # Randomly create bias #####################################################
    bias_idx = np.random.randint(len(feature_names))
    bias_word = feature_names[bias_idx]

    train_df['label_bias'] = train_df['label_orig']
    bias_example_mask = train_df.iloc[:, bias_idx] > 0
    train_df.loc[bias_example_mask, 'label_bias'] = 0

    test_df['label_bias'] = test_df['label_orig']
    bias_example_mask = test_df.iloc[:, bias_idx] > 0
    test_df.loc[bias_example_mask, 'label_bias'] = 0

    print('\tBIAS_WORD = "{}"'.format(bias_word))
    runlog['bias_word'] = bias_word

    # Training unbiased and biased model #######################################
    X_train = train_df.drop(['label_orig', 'label_bias'], axis=1).values
    y_train_orig = train_df['label_orig'].values
    y_train_bias = train_df['label_bias'].values

    # NOTE: Saving a name for the classifier in the runlog dict is crucial
    # TODO: make selecting a model a cmd line arg

    model_orig = RandomForestClassifier(n_estimators=50)
    model_bias = RandomForestClassifier(n_estimators=50)
    runlog['model_type'] = 'RandomForestClassifier'

    # model_orig = LogisticRegression()
    # model_bias = LogisticRegression()
    # runlog['model_type'] = 'LogisticRegression'

    # model_orig = MLPClassifier()
    # model_bias = MLPClassifier()
    # runlog['model_type'] = 'MLPClassifier'

    orig_start = time.time()
    model_orig.fit(X_train, y_train_orig)
    orig_end = time.time()

    bias_start = time.time()
    model_bias.fit(X_train, y_train_bias)
    bias_end = time.time()

    print('\tMODEL_TYPE = {}'.format(runlog['model_type']))
    print('\tORIG_TRAIN_TIME = {:.2f} sec.'.format(orig_end - orig_start))
    print('\tBIAS_TRAIN_TIME = {:.2f} sec.'.format(bias_end - bias_start))

    # Evaluate both models on biased region R and ~R ###########################
    bias_example_mask = test_df.iloc[:, bias_idx] > 0
    R = test_df[bias_example_mask]
    not_R = test_df[~bias_example_mask]

    runlog['results'] = evaluate_models(model_orig, model_bias, R, not_R)

    ############################################################################
    # Explainers ###############################################################
    ############################################################################

    # Number of features the explainer is allowed to return
    for n_features in range(1, 11):
        runlog['n_features'] = n_features

        # Sepatate R from the test set ########################################
        X_test_R = R.drop(['label_orig', 'label_bias'], axis=1).values
        y_test_R = R['label_bias'].values

        assert(np.all(X_test_R[:, bias_idx] > 0))

        # Initialize explainers ##############################################

        # LIME Tabular Explainer
        lime_explainer = LimeTabular(
                predict_fn=model_bias.predict_proba,
                data=X_train,
                feature_names=feature_names,
                random_state=seed,
                explain_kwargs={'num_features': n_features}
        )


        # SHAP Kernel Explainer
        background_vals = np.median(X_train, axis=0).reshape(1, -1)
        shap_explainer = ShapKernel(
                predict_fn=model_bias.predict_proba,
                data=background_vals,
                feature_names=feature_names,
                random_state=seed
        )

        explainers = {
                "LIME": lime_explainer,
                "SHAP": shap_explainer,
        }

        # Test and compare the explainers ##################################

        # Test explainers on a subsample of the test set
        n_samples = min(100, X_test_R.shape[0])
        X_explain, y_explain = resample(
                X_test_R,
                y_test_R,
                replace=False,
                n_samples=n_samples
        )

        # Generating Explanations #########################################

        for name in explainers:
            print('\tEXPLAINER = {}'.format(name))
            print('\t\tNUM_SAMPLES = {}'.format(n_samples))
            runlog['explainer'] = name
            runlog['n_samples'] = n_samples

            explainer = explainers[name]
            exp = explainer.explain_local(X_explain, y_explain, name=name)

            recall = 0
            for i in range(n_samples):
                # Getting important features and computing recall

                names = exp.data(i)['names']
                values = exp.data(i)['scores']

                # This call returns a ranked list of the most important features
                sorted_pairs = sorted(
                        zip(names, values),
                        key = lambda x : abs(x[1]),
                        reverse=True
                )

                important_features, _ = zip(*sorted_pairs[:n_features])

                print('\t\tIMPORTANT_FEATURES = {}'.format(important_features))

                # TODO: Chhange this to work generally (multiple bias words)
                if bias_word in important_features:
                    recall += 1

            recall /= n_samples
            print('\t\tRECALL = {}'.format(recall))
            runlog['recall'] = recall

            # save log ########################################################
            print('Saving log')
            if LOGGING_ENABLED:
                print('Logging')
                log_dir = os.path.join(args.log_dir, dataset_name,
                        runlog['model_type'])
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                log_path = os.path.join(log_dir, '{0}_{1}_{2:04d}.json'.format(
                    runlog['explainer'],
                    runlog['n_features'],
                    runlog['seed']))
                print('writing log to: {}'.format(log_path))
                with open(log_path, 'w') as f:
                    json.dump(runlog, f)


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    Pool(4).map(run_seed, range(args.seed_low, args.seed_high))
