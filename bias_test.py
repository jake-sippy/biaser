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
from sklearn.feature_extraction.text import CountVectorizer

# Random seed, to be replaced by loops later
SEED = 0

# The minimum occurance of words to include, currently just chosen ad hoc
MIN_OCCURANCE = 1000

# Path to the dataset to load in
DATASET_PATH = 'data/reviews_Home_and_Kitchen/reviews.json'


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
        labels.append(json_line['label'])

# # Class balance
# print('Class balance:')
# unique_labels, label_counts = np.unique(labels, return_counts=True)
# total_count = sum(label_counts)
# for label, count in zip(unique_labels, label_counts):
#     print('{0}: {1:.2f}'.format(label, count / total_count))


print('\nGetting word counts...')
vectorizer = CountVectorizer(min_df=MIN_OCCURANCE)
X = vectorizer.fit_transform(corpus)

# this is the number of words that occur in the corpus > MIN_OCCURANCE
possible_words = X.shape[1]
print('\nThere are {} words in the corpus with greater than {} occurances.'
        .format(possible_words, MIN_OCCURANCE))

bias_word_idx = np.random.randint(possible_words)
bias_word = vectorizer.get_feature_names()[bias_word_idx]

print('The random word selected was "{}" with {} occurances.'
        .format(bias_word, X[:, bias_word_idx].sum()))

