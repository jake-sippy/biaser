import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        recall_score,
        precision_score)
from sklearn.model_selection import train_test_split

# Load the dataset from the given path and returned the split, un-pre-processed
# version.
def load_dataset(data_path, train_size, runlog):
    dataset_name = data_path.split('/')[-1].split('.csv')[0]
    runlog['dataset'] = dataset_name
    runlog['train_size'] = train_size
    print('Loading dataset...')
    print('\tDATASET = {}'.format(dataset_name))
    print('\tTRAIN_SIZE = {}'.format(train_size))
    data = pd.read_csv(data_path, header=None, names=['reviews', 'labels'])
    reviews = data['reviews'].astype(str).values
    labels = data['labels'].values
    print('\tNUM_SAMPLES = {}'.format(len(data)))
    test_size = 1.0 - train_size
    return train_test_split(
            reviews,
            labels,
            train_size=train_size,
            test_size=test_size)


# Helper for resample
def oversample(X, y):
    data = pd.DataFrame(data=zip(X, y), columns=['X', 'y'])
    counts = data.y.value_counts()
    assert len(counts) == 2, "This method only written for binary classes"
    small_class = data[ data.y == counts.idxmin() ]
    large_class = data[ data.y == counts.idxmax() ]
    oversampled = small_class.sample(counts.max(), replace=True)
    data = pd.concat([oversampled, large_class], axis=0)
    return data.X.values, data.y.values


def train_models(pipeline, train_df, runlog):
    print('Training unbiased and biased models...')
    print('\tMODEL_TYPE = {}'.format(runlog['model_type']))

    # Calculate sample weights
    r = train_df['biased'].sum()
    nr = len(train_df) - r
    print('\t R:', r / (r + nr))
    print('\t~R:', nr / (r + nr))
    sample_weight_orig = []
    sample_weight_bias = []
    for biased in train_df['biased'].values:
        sample_weight_orig.append(1.0)
        sample_weight_bias.append(1.0 if biased else r / nr)
        # sample_weight_bias.append(nr / r if biased else 1.0)

    # Actual training
    X_train = train_df['reviews'].values
    X_train_biased = train_df['biased'].values
    y_train_orig = train_df['label_orig'].values
    y_train_bias = train_df['label_bias'].values

    skorch = runlog['model_type'] in ['dnn', 'lstm']

    pipe_orig = pipeline
    pipe_bias = clone(pipeline)
    if skorch:
        model_orig = pipe_orig.steps.pop(-1)
        model_bias = pipe_bias.steps.pop(-1)
        pipe_orig.fit(X_train)
        pipe_bias.fit(X_train)
        X_train = pipe_orig.transform(X_train)
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

    # Unbiased model
    print('Training unbiased model...')
    start = time.time()
    if skorch:
        model_orig[1].fit(X_train_orig, y_train_orig)
        pipe_orig.steps.append(model_orig)
    else:
        sample_weight_orig = [1.0] * len(X_train)
        pipe_orig.fit(X_train, y_train_orig,
                model__sample_weight=sample_weight_orig)
    end = time.time()
    print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))

    # Biased model
    print('Training biased model...')
    start = time.time()
    if skorch:
        model_bias[1].fit(X_train_bias, y_train_bias)
        pipe_bias.steps.append(model_bias)
    else:
        pipe_bias.fit(X_train, y_train_bias,
                model__sample_weight=sample_weight_bias)
    end = time.time()
    print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))

    return pipe_orig, pipe_bias


# Split test data into R and ~R and compute the accruacies of the two models
def evaluate_models(model_orig, model_bias, test_df, runlog):
    print('Evaluating unbiased and biased models on test set...')
    mask = test_df['biased']
    test_r = test_df[mask]
    test_nr = test_df[~mask]

    X_r = test_r['reviews'].values
    X_nr = test_nr['reviews'].values

    # Get original model's accuracy on R and ~R
    y_r = test_r['label_orig'].values
    y_nr = test_nr['label_orig'].values
    orig_r_acc = accuracy_score(y_r, model_orig.predict(X_r))
    orig_nr_acc = accuracy_score(y_nr, model_orig.predict(X_nr))

    # Get biased model's accuracy on R and ~R
    y_r = test_r['label_bias'].values
    y_nr = test_nr['label_bias'].values

    pred_r = model_bias.predict(X_r)
    bias_r_acc = accuracy_score(y_r, pred_r)
    bias_nr_acc = accuracy_score(y_nr, model_bias.predict(X_nr))

    print('\t               R       !R')
    print('\torig model | {0:.3f} | {1:.3f}'.format(orig_r_acc, orig_nr_acc))
    print('\tbias model | {0:.3f} | {1:.3f}'.format(bias_r_acc, bias_nr_acc))
    runlog['results'] = [[orig_r_acc, orig_nr_acc], [bias_r_acc, bias_nr_acc]]


# Split test data into R and ~R and compute the accruacies of the two models
def evaluate_models_test(model_orig, model_bias, test_df, runlog):
    print('Evaluating unbiased and biased models on test set...')
    X = test_df['reviews'].values
    y_orig = test_df['label_orig'].values
    y_bias = test_df['label_bias'].values

    # Get original model's accuracy on R and ~R
    y_pred_orig = model_orig.predict(X)
    y_pred_bias = model_bias.predict(X)
    runlog['orig_test_acc'] = accuracy_score(y_orig, y_pred_orig)
    runlog['bias_test_acc'] = accuracy_score(y_bias, y_pred_bias)
    runlog['orig_test_class_report'] = classification_report(y_orig, y_pred_orig)
    runlog['bias_test_class_report'] = classification_report(y_bias, y_pred_bias)

    print('\torig model accuracy:', runlog['orig_test_acc'])
    print('\torig model classification_report:\n',
            runlog['orig_test_class_report'])
    print('\tbias model accuracy:', runlog['bias_test_acc'])
    print('\tbias model classification_report:\n',
            runlog['bias_test_class_report'])

def save_log(log_dir, filename, runlog):
    log_dir = os.path.join(
            log_dir,
            runlog['test_name'],
            runlog['dataset'],
            runlog['model_type'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, filename)
    print('Writing log to: {}'.format(log_path))
    with open(log_path, 'w') as f:
        json.dump(runlog, f)

