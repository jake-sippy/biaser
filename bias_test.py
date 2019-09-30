# This module is to test if biasing the dataset in a simple way has
# the effect on model performances that we would hope to see.
#
# Reviews at this stage are passed in as lines of json,
# each line is one review of the form:
# {"text": ..., "label": ...}

import json
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        f1_score
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

# Random seed, to be replaced by loops later
SEED = 0

# The minimum occurance of words to include, currently just chosen ad hoc
MIN_OCCURANCE = 1000

# Path to the dataset to load in
DATASET_PATH = 'data/reviews_Musical_Instruments/reviews.json'

# Ratio to split for the train set (including dev)
TRAIN_SIZE = 0.8


def load_dataset(path):
    print('\nLoading dataset from: {} ...'.format(path))

    with open(path, 'rb') as f:
        lines = f.readlines()
        corpus = []
        labels = []

        for line in tqdm(lines):
            json_line = json.loads(line)
            corpus.append(json_line['text'])
            labels.append(1 if json_line['label'] == 'positive' else 0)

    return corpus, labels


def vectorize_dataset(corpus, labels, min_df):
    print('\nConverting dataset to bag-of words vector representation...')
    vectorizer = CountVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(corpus).toarray()
    y = np.array(labels)
    feature_names = vectorizer.get_feature_names()
    data = pd.DataFrame(data=X, columns=feature_names)
    data['label'] = y
    return data


def oversample(df):
    print('\nOversampling smaller class to correct imbalance...')
    counts = df.label.value_counts()
    smaller_class = df[ df['label'] == counts.idxmin() ]
    larger_class = df[ df['label'] == counts.idxmax() ]
    over = smaller_class.sample(counts.max(), replace=True)
    return pd.concat([over, larger_class], axis=0)


def evaluate_models(model_orig, model_bias, r, not_r):
    X_r = r.drop('label', axis=1).values
    y_r = r['label'].values

    X_not_r = not_r.drop('label', axis=1).values
    y_not_r = not_r['label'].values

    pred_orig_r = model_orig.predict(X_r)
    pred_orig_not_r = model_orig.predict(X_not_r)

    pred_bias_r = model_bias.predict(X_r)
    pred_bias_not_r = model_bias.predict(X_not_r)

    print('             R      !R')

    print('orig model | {0:3.2f} | {1:3.2f}'
            .format(
                accuracy_score(y_r, pred_orig_r),
                accuracy_score(y_not_r, pred_orig_not_r)))

    print('bias model | {0:3.2f} | {1:3.2f}'
            .format(
                accuracy_score(y_r, pred_bias_r),
                accuracy_score(y_not_r, pred_bias_not_r)))


if __name__ == '__main__':
    np.random.seed(SEED)
    print('Running bias test with seed={}. Not logging.'.format(SEED))

    # get data
    corpus, labels = load_dataset(DATASET_PATH)
    data = vectorize_dataset(corpus, labels, MIN_OCCURANCE)

    # split
    train_df, test_df = train_test_split(data, train_size=0.80)

    # oversample to correct class imbalance
    train_df = oversample(train_df)

    # introduce bias
    n_examples, n_features = train_df.drop('label', axis=1).shape
    bias_word_idx = np.random.randint(n_features)
    print('Word selected for bias was "{}"'
            .format(train_df.columns[bias_word_idx]))

    train_df['label_bias'] = train_df['label']
    mask = train_df.iloc[:, bias_word_idx] > 0
    train_df.loc[mask, 'label_bias'] = 0

    # training models
    X_train = train_df.drop(['label', 'label_bias'], axis=1).values
    y_train_orig = train_df['label'].values
    y_train_bias = train_df['label_bias'].values

    model_orig = RandomForestClassifier(n_estimators=100)
    model_bias = RandomForestClassifier(n_estimators=100)

    model_orig.fit(X_train, y_train_orig)
    model_bias.fit(X_train, y_train_bias)

    # evaluate both true test R and true test !R

    mask = test_df.iloc[:, bias_word_idx] > 0
    R = test_df[mask]
    not_R = test_df[~mask]

    evaluate_models(model_orig, model_bias, R, not_R)
