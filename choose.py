import os
import time
import json
import pydot
import torch
import argparse
import numpy as np
from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        export_graphviz,
        _tree
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_data(filename):
    reviews = []
    scores = []
    with open(filename, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            reviews.append(json_line['text'])
            scores.append(json_line['label'])
    return reviews, scores


def get_ngrams(tree, max_depth):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    result = [""] * max_depth

    def recurse(node, depth):
        if depth <= max_depth:
            name = feature_name[node]
            result[depth - 1] = name
            threshold = tree_.threshold[node]
            recurse(tree_.children_right[node], depth + 1)
    recurse(0, 1)
    print(result)


depth = 5
reviews, scores = load_data('data/reviews_Video_Games/train.json')

print('Splitting data...')
(x_train,
 x_dev,
 y_train,
 y_dev) = train_test_split(reviews, scores, train_size=0.8, test_size=0.2)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression(penalty='l2',
                               solver='lbfgs'))
    # ('clf', DecisionTreeClassifier(max_depth=depth, min_samples_leaf=500))
])

# get feature names for better visualization
vocab = pipeline.named_steps['vect'].vocabulary_
feature_names = [''] * len(vocab)
print('vocab length:', len(vocab))
for key in vocab:
    feature_names[vocab[key]] = key

print('Training Tree...')
pipeline.fit(x_train, y_train)

clf = pipeline.named_steps['clf']
weights = clf.coef_.flatten()
print(weights)
print(weights.shape)
start = time.time()
idx = np.argsort(weights)[:10]
end = time.time()
print(weights[idx])
print(feature_names[idx])
print(end - start)

# get_ngrams(clf, depth)

# print('Visualizing Tree...')
# export_graphviz(pipeline.named_steps['tree'], 'tree.dot',
#                 feature_names=feature_names,
#                 class_names=['negative', 'positive'])
# (graph,) = pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')
#
# print('Scoring Tree...')
# y_pred = pipeline.predict(x_dev)
# print(classification_report(y_dev, y_pred))
# print("Accuracy: %.2f %%" % (accuracy_score(y_dev, y_pred) * 100.0))
