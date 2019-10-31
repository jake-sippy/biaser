# Utils for loading and handling data that is shared between tests. These utils
# also take the runlog dictionary object and store useful metadata

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

# Get a more readable name for the datset from the filename of the cleaned data
def get_dataset(data_path, train_size, runlog):
    dataset_name = data_path.split('/')[-1].split('.csv')[0]
    runlog['dataset'] = dataset_name
    print('\tDATASET = {}'.format(dataset_name))
    data = pd.read_csv(data_path, header=None, names=['reviews', 'labels'])
    print('Splitting dataset...')
    print('\tTRAIN_SIZE = {}'.format(train_size))
    reviews = data['reviews'].astype(str).values
    labels = data['labels'].values
    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = train_test_split(reviews, labels, train_size=train_size)
    runlog['train_size'] = train_size
    return reviews_train, reviews_test, labels_train, labels_test


# Convert text dataset to vectorized representation
def vectorize_dataset(reviews_train, reviews_test, labels_train, labels_test,
        min_occur, max_occur, runlog):
    print('Converting text dataset to vector representation...')
    print('\tMIN_OCCURANCE = {}'.format(min_occur))
    print('\tMAX_OCCURANCE = {}'.format(max_occur))
    pipeline = Pipeline(steps=[
        ('vectorizer', CountVectorizer(min_df=min_occur, max_df=max_occur)),
        ('scaler', StandardScaler(with_mean=False))
    ])

    X_train = pipeline.fit_transform(reviews_train).toarray()
    y_train = np.array(labels_train)
    X_test = pipeline.transform(reviews_test).toarray()
    y_test = np.array(labels_test)
    feature_names = pipeline.named_steps['vectorizer'].get_feature_names()
    print('\tFEATURES EXTRACTED = {}'.format(len(feature_names)))

    runlog['min_occur'] = min_occur
    runlog['max_occur'] = max_occur

    return X_train, X_test, y_train, y_test, pipeline, feature_names


# Helper for resample
def oversample(df):
    counts = df.label.value_counts()
    smaller_class = df[ df['label'] == counts.idxmin() ]
    larger_class = df[ df['label'] == counts.idxmax() ]
    over = smaller_class.sample(counts.max(), replace=True)
    return pd.concat([over, larger_class], axis=0)


# Resample the training data to have equal class balance
def resample(X_train, y_train, feature_names):
    print('Resampling to correct class imbalance...')
    train_df = pd.DataFrame(data=X_train, columns=feature_names)
    train_df['label'] = y_train

    value_counts = train_df.label.value_counts()

    print('\tORIGINAL BALANCE = ')
    print('\t\tClass_0 = {}\n\t\tClass_1 = {}'
            .format(value_counts[0], value_counts[1]))

    train_df = oversample(train_df)

    value_counts = train_df.label.value_counts()
    print('\tCORRECTED BALANCE = ')
    print('\t\tClass_0 = {}\n\t\tClass_1 = {}'
            .format(value_counts[0], value_counts[1]))
    return train_df


def create_bias(train_df, feature_names, runlog):
    print('Randomly selecting word to bias...')
    bias_idx = np.random.randint(len(train_df.columns))
    bias_word = feature_names[bias_idx]
    print('\tBIAS_WORD = "{}"'.format(bias_word))
    runlog['bias_word'] = bias_word
    train_df['label_bias'] = train_df['label']
    mask = train_df.iloc[:, bias_idx] > 0
    train_df.loc[mask, 'label_bias'] = 0
    return bias_idx, bias_word


def train_models(model_type, models, train_df, runlog):
    print('Training models...')
    print('\tMODEL_TYPE = {}'.format(model_type))
    runlog['model_type'] = model_type

    X_train = train_df.drop(['label', 'label_bias'], axis=1).values
    y_train_orig = train_df['label'].values
    y_train_bias = train_df['label_bias'].values

    model_orig = clone(models[model_type])
    model_bias = clone(models[model_type])

    print('Training unbiased model...')
    start = time.time()
    model_orig.fit(X_train, y_train_orig)
    end = time.time()
    print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))

    print('Training biased model...')
    start = time.time()
    model_bias.fit(X_train, y_train_bias)
    end = time.time()
    print('\tTRAIN_TIME = {:.2f} sec.'.format(end - start))
    return model_orig, model_bias


# Split test data into R and ~R and compute the accruacies of the two models
def evaluate_models(model_orig, model_bias, test_df, bias_idx, runlog):
    print('Evaluating unbiased and biased models on test set...')
    mask = test_df.iloc[:, bias_idx] > 0
    r = test_df[mask]
    not_r = test_df[~mask]

    X_r = r.drop('label', axis=1).values
    y_r = r['label'].values

    X_not_r = not_r.drop('label', axis=1).values
    y_not_r = not_r['label'].values

    pred_orig_r = model_orig.predict(X_r)
    pred_orig_not_r = model_orig.predict(X_not_r)

    pred_bias_r = model_bias.predict(X_r)
    pred_bias_not_r = model_bias.predict(X_not_r)

    orig_r = accuracy_score(y_r, pred_orig_r)
    orig_not_r = accuracy_score(y_not_r, pred_orig_not_r)

    bias_r = accuracy_score(y_r, pred_bias_r)
    bias_not_r = accuracy_score(y_not_r, pred_bias_not_r)

    print('\t              R     !R')
    print('\torig model | {0:3.2f} | {1:3.2f}'.format(orig_r, orig_not_r))
    print('\tbias model | {0:3.2f} | {1:3.2f}'.format(bias_r, bias_not_r))
    runlog['results'] = [[orig_r, orig_not_r], [bias_r, bias_not_r]]


def save_log(log_dir, runlog):
    log_dir = os.path.join(
            log_dir,
            runlog['dataset'],
            runlog['model_type']
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(
            log_dir,
            '{0:04d}.json'.format(runlog['seed'])
    )

    print('Writing log to: {}'.format(log_path))
    with open(log_path, 'w') as f:
        json.dump(runlog, f)

