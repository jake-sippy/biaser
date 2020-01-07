# Utils for loading and handling data that is shared between tests. These utils
# also usually take the runlog dictionary object and store useful metadata

import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from interpret.blackbox import (
#         PartialDependence,
#         ShapKernel,
#         LimeTabular,
#         MorrisSensitivity
# )


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
    print('\tNUM_SAMPLES = {}'.format(len(reviews)))
    return train_test_split(reviews, labels, train_size=train_size)


# Convert text dataset to vectorized representation
def vectorize_dataset(reviews_train, reviews_test, min_occur, max_occur, runlog):
    print('Converting text dataset to vector representation...')
    print('\tMIN_OCCURANCE = {}'.format(min_occur))
    print('\tMAX_OCCURANCE = {}'.format(max_occur))
    pipeline = Pipeline(steps=[
        ('vectorizer', CountVectorizer(
            min_df=min_occur,
            max_df=max_occur,
            binary=True
        )),
        # ('scaler', StandardScaler(with_mean=False))
    ])

    X_train = pipeline.fit_transform(reviews_train).toarray()
    X_test = pipeline.transform(reviews_test).toarray()
    feature_names = pipeline.named_steps['vectorizer'].get_feature_names()
    # DEBUG print doc frequencies
    # occurances = zip(feature_names, np.mean(X_train, axis=1))
    # for feat, occur in occurances:
    #     print('{}:\t\t{}'.format(feat, occur))
    # exit()
    print('\tFEATURES EXTRACTED = {}'.format(len(feature_names)))

    runlog['min_occur'] = min_occur
    runlog['max_occur'] = max_occur

    return X_train, X_test, pipeline, feature_names


# Helper for resample
def oversample(df):
    counts = df.label_bias.value_counts()
    smaller_class = df[ df.label_bias == counts.idxmin() ]
    larger_class = df[ df.label_bias == counts.idxmax() ]
    over = smaller_class.sample(counts.max(), replace=True)
    return pd.concat([over, larger_class], axis=0)


# Resample the training data to have equal class balance
def resample(train_df, feature_names):
    print('Resampling to correct class imbalance...')
    value_counts = train_df['label_bias'].value_counts(normalize=True)
    orig_balance = value_counts.sort_index().values

    print('\tORIGINAL BALANCE = ')
    print('\t\tClass_0 = {0:.2f}\n\t\tClass_1 = {1:.2f}'.format(
        orig_balance[0], orig_balance[1]))

    train_df = oversample(train_df)

    value_counts = train_df['label_bias'].value_counts(normalize=True)
    new_balance = value_counts.sort_index().values
    print('\tCORRECTED BALANCE = ')
    print('\t\tClass_0 = {0:.2f}\n\t\tClass_1 = {1:.2f}'.format(
        new_balance[0], new_balance[1]))
    return train_df


# TODO remove
def create_bias(train_df, test_df, feature_names, balance, runlog, features=1):
    print('Old bias method being called, please remove')
    print(train_df.head())
    exit()


def train_models(
        model_type,
        models,
        train_df,
        runlog,
        bias_only=False,    # Only return a biased model
):
    print('Training model(s)...')
    print('\tMODEL_TYPE = {}'.format(model_type))
    runlog['model_type'] = model_type

    biased = train_df['biased'].values
    r = 0       # region R
    nr = 0      # region not R
    for b in biased:
        if b:
            r += 1
        else:
            nr += 1

    sample_weights = []
    for b in biased:
        sample_weights.append(3 if b else r/nr)

    # print('Region R balance: {}'.format(r/(r+nr)))

    X_train = train_df.drop(['label_orig', 'label_bias', 'biased'], axis=1).values
    y_train_orig = train_df['label_orig'].values
    y_train_bias = train_df['label_bias'].values

    model_orig = clone(models[model_type])
    model_bias = clone(models[model_type])

    if not bias_only:
        print('Training unbiased model...')
        start = time.time()
        model_orig.fit(X_train, y_train_orig)
        end = time.time()
        print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))

    print('Training biased model...')
    start = time.time()
    model_bias.fit(X_train, y_train_bias, sample_weight=sample_weights)
    end = time.time()
    print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))

    if bias_only:
        return model_bias
    else:
        return model_orig, model_bias


# Split test data into R and ~R and compute the accruacies of the two models
def evaluate_models(model_orig, model_bias, test_df, runlog):
    print('Evaluating unbiased and biased models on test set...')
    mask = test_df['biased']
    test_r = test_df[mask]
    test_nr = test_df[~mask]

    drop_cols = ['label_orig', 'label_bias', 'biased']
    X_r = test_r.drop(drop_cols, axis=1).values
    X_nr = test_nr.drop(drop_cols, axis=1).values

    # Get original model's accuracy on R and ~R
    y_r = test_r['label_orig'].values
    orig_r_acc = accuracy_score(y_r, model_orig.predict(X_r))
    y_nr = test_nr['label_orig'].values
    orig_nr_acc = accuracy_score(y_nr, model_orig.predict(X_nr))

    # Get biased model's accuracy on R and ~R
    y_r = test_r['label_bias'].values
    bias_r_acc = accuracy_score(y_r, model_bias.predict(X_r))
    y_nr = test_nr['label_bias'].values
    bias_nr_acc = accuracy_score(y_nr, model_bias.predict(X_nr))

    print('\t               R       !R')
    print('\torig model | {0:.3f} | {1:.3f}'.format(orig_r_acc, orig_nr_acc))
    print('\tbias model | {0:.3f} | {1:.3f}'.format(bias_r_acc, bias_nr_acc))
    runlog['results'] = [[orig_r_acc, orig_nr_acc], [bias_r_acc, bias_nr_acc]]


def save_log(log_dir, filename, runlog):
    log_dir = os.path.join(
            log_dir,
            runlog['test_name'],
            runlog['dataset'],
            runlog['model_type']
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, filename)

    print('Writing log to: {}'.format(log_path))
    with open(log_path, 'w') as f:
        json.dump(runlog, f)

