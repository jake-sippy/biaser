# This module is to test if biasing the dataset in a simple way has
# the effect on model performances that we would hope to see.
#
# Reviews at this stage are passed in as lines of json,
# each line is one review of the form:
# {"text": ..., "label": ...}

import json
from tqdm import tqdm
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Random seed, to be replaced by loops later
SEED = 0

# The minimum occurance of words to include, currently just chosen ad hoc
MIN_OCCURANCE = 1000

# Path to the dataset to load in
DATASET_PATH = 'data/reviews_Musical_Instruments/reviews.json'

# Ratio to split for the train set (including dev)
TRAIN_SIZE = 0.8


np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


print('Running bias test. Not logging.')

# Loading dataset to correct form
print('\nLoading dataset: {}'.format(DATASET_PATH))

with open(DATASET_PATH, 'rb') as f:
    lines = f.readlines()
    corpus = []
    labels = []

    for line in tqdm(lines):
        json_line = json.loads(line)
        corpus.append(json_line['text'])
        labels.append(1 if json_line['label'] == 'positive' else 0)

# # Class balance
# print('Class balance:')
# unique_labels, label_counts = np.unique(labels, return_counts=True)
# total_count = sum(label_counts)
# for label, count in zip(unique_labels, label_counts):
#     print('{0}: {1:.2f}'.format(label, count / total_count))


print('\nGetting word counts...')
vectorizer = CountVectorizer(min_df=MIN_OCCURANCE)
X = vectorizer.fit_transform(corpus)
y = np.array(labels)

# this is the number of words that occur in the corpus > MIN_OCCURANCE
possible_words = X.shape[1]
print('\nThere are {} words in the corpus with greater than {} occurances.'
        .format(possible_words, MIN_OCCURANCE))

bias_word_idx = np.random.randint(possible_words)
bias_word = vectorizer.get_feature_names()[bias_word_idx]

print('The random word selected was "{0}" (index {1}) which occurs in {2:.2f}% of reviews.'
        .format(
            bias_word,
            bias_word_idx,
            100 * X[:, bias_word_idx].count_nonzero() / X.shape[0]))


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

bias_reviews = []
for i, row in enumerate(X_train):
    if row[0, bias_word_idx] > 0:
        bias_reviews.append(i)

print('\nBiasing {} of {} total reivews'.format(
    len(bias_reviews), X_train.shape[0]))
y_train_bias = y_train[:]

pos = sum(y_train_bias[bias_reviews])
neg = len(y_train_bias[bias_reviews]) - pos
