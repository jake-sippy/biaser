import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        recall_score,
        precision_score
)


# Load the dataset from the given path and returned the split
def load_dataset(data_path, train_size, runlog, quiet=False):
    dataset_name = data_path.split('/')[-1].split('.csv')[0]
    runlog['dataset'] = dataset_name
    runlog['train_size'] = train_size
    data = pd.read_csv(data_path, header=None, names=['reviews', 'labels'])
    reviews = data['reviews'].astype(str).values
    labels = data['labels'].values
    counts = data['labels'].value_counts()
    if not quiet: print('Loading dataset...')
    if not quiet: print('\tDATASET = {}'.format(dataset_name))
    if not quiet: print('\tTRAIN_SIZE = {}'.format(train_size))
    if not quiet: print('\tNUM_SAMPLES = {}'.format(len(data)))
    if not quiet: print('\t% POSITIVE = {:.2f}'.format(counts[1] / (counts[0] + counts[1])))
    return train_test_split(reviews, labels, train_size=train_size,
            test_size=1-train_size)


def train_models(model_constructor, train_df, runlog, bias_only=False, quiet=False):
    if not quiet: print('Training unbiased and biased models...')
    if not quiet: print('\tMODEL_TYPE = {}'.format(runlog['model_type']))

    # Calculate sample weights
    r = train_df['biased'].sum()
    nr = len(train_df) - r
    if not quiet: print('\t R:', r / (r + nr))
    if not quiet: print('\t~R:', nr / (r + nr))
    sample_weight_orig = []
    sample_weight_bias = []
    for biased in train_df['biased'].values:
        sample_weight_orig.append(1.0)
        sample_weight_bias.append(1.0 if biased else r / nr)

    # Actual training
    X_train = train_df['reviews'].values
    X_train_biased = train_df['biased'].values
    y_train_orig = train_df['label_orig'].values
    y_train_bias = train_df['label_bias'].values

    if not bias_only:
        pipe_orig = model_constructor()
    pipe_bias = model_constructor()

    if runlog['model_type'] in ['mlp', 'lstm']:
        if not bias_only:
            model_orig = pipe_orig.steps.pop(-1)
            pipe_orig.fit(X_train)

        model_bias = pipe_bias.steps.pop(-1)
        pipe_bias.fit(X_train)
        X_train = pipe_bias.transform(X_train)
        X_train_bias = {
            'data': X_train,
            'biased': X_train_biased,
            'sample_weight': sample_weight_orig
        }

        X_train_orig = {
            'data': X_train,
            'biased': X_train_biased,
            'sample_weight': sample_weight_bias
        }

        if not bias_only:
            if not quiet: print('Training unbiased model...')
            model_orig[1].fit(X_train_orig, y_train_orig)
            pipe_orig.steps.append(model_orig)
        if not quiet: print('Training biased model...')
        model_bias[1].fit(X_train_bias, y_train_bias)
        pipe_bias.steps.append(model_bias)
    else:
        if not bias_only:
            if not quiet: print('Training unbiased model...')
            pipe_orig.fit(X_train, y_train_orig,
                    model__sample_weight=sample_weight_orig)
        if not quiet: print('Training biased model...')
        pipe_bias.fit(X_train, y_train_bias,
                model__sample_weight=sample_weight_bias)

    if bias_only:
        return pipe_bias
    else:
        return pipe_orig, pipe_bias


# Split test data into R and ~R and compute the accruacies of the two models
def evaluate_models(model_orig, model_bias, test_df, runlog, quiet=False):
    if not quiet: print('Evaluating unbiased and biased models on test set...')
    mask = test_df['biased']
    test_r = test_df[mask]
    test_nr = test_df[~mask]

    X_r = test_r['reviews'].values
    X_nr = test_nr['reviews'].values

    # Get original model's accuracy on R and ~R
    y_r = test_r['label_orig'].values
    y_nr = test_nr['label_orig'].values
    pred_r = model_orig.predict(X_r)
    pred_nr = model_orig.predict(X_nr)
    print(y_r)
    print(pred_r)
    orig_r_acc = accuracy_score(y_r, pred_r)
    orig_nr_acc = accuracy_score(y_nr, pred_nr)

    # Get biased model's accuracy on R and ~R
    y_r = test_r['label_bias'].values
    y_nr = test_nr['label_bias'].values
    pred_r = model_bias.predict(X_r)
    pred_nr = model_bias.predict(X_nr)
    bias_r_acc = accuracy_score(y_r, pred_r)
    bias_nr_acc = accuracy_score(y_nr, pred_nr)

    if not quiet: print('\t               R       !R')
    if not quiet: print('\torig model | {0:.3f} | {1:.3f}'.format(orig_r_acc, orig_nr_acc))
    if not quiet: print('\tbias model | {0:.3f} | {1:.3f}'.format(bias_r_acc, bias_nr_acc))
    runlog['results'] = [[orig_r_acc, orig_nr_acc], [bias_r_acc, bias_nr_acc]]


# Split test data into R and ~R and compute the accruacies of the two models
def evaluate_models_test(model_orig, model_bias, test_df, runlog, quiet=False):
    if not quiet: print('Evaluating unbiased and biased models on test set...')
    X = test_df['reviews'].values
    y_orig = test_df['label_orig'].values
    y_bias = test_df['label_bias'].values

    # Get original model's accuracy on R and ~R
    y_pred_orig = model_orig.predict(X)
    y_pred_bias = model_bias.predict(X)
    runlog['orig_test_acc'] = accuracy_score(y_orig, y_pred_orig)
    runlog['bias_test_acc'] = accuracy_score(y_bias, y_pred_bias)
    runlog['orig_test_f1'] = f1_score(y_orig, y_pred_orig)
    runlog['bias_test_f1'] = f1_score(y_bias, y_pred_bias)

    if not quiet: print('\torig model accuracy:', runlog['orig_test_acc'])
    if not quiet: print('\torig model f1:', runlog['orig_test_f1'])
    if not quiet: print('\tbias model accuracy:', runlog['bias_test_acc'])
    if not quiet: print('\tbias model f1:', runlog['bias_test_f1'])


def save_log(log_dir, runlog, quiet=False):

    if runlog['test_name'] == 'budget_test':
        filename = '{:s}_{:d}_{:03d}_{:02d}_{:03d}.json'.format(
                runlog['explainer'],
                runlog['bias_len'],
                runlog['seed'],
                runlog['budget'],
                runlog['example_id'])

    directory = os.path.join(
            log_dir,
            runlog['test_name'],
            runlog['dataset'],
            runlog['model_type'])

    if runlog['toy']:
        # Save in 'toy' sub-directory
        directory = os.path.join(
                log_dir, 'toy',
                runlog['test_name'],
                runlog['dataset'],
                runlog['model_type'])

    if not os.path.exists(directory):
        os.makedirs(directory)

    log_path = os.path.join(directory, filename)
    if not quiet: print('Writing log to: {}'.format(log_path))
    with open(log_path, 'w') as f:
        json.dump(runlog, f)

