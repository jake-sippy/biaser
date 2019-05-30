import os
import json
import pydot
import torch
import argparse
import numpy as np
from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        export_graphviz
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Train and visualize a decision tree',
    )
    parser.add_argument('path', metavar='path',
                        help='Path of dataset to train DT on')
    parser.add_argument('-d', '--depth', metavar='N', type=int, default=3,
                        help='Max depth of decision tree (default = 3)')
    args = parser.parse_args()
    reviews, scores = load_data(args.path)

    print('Splitting data...')
    (x_train,
     x_dev,
     y_train,
     y_dev) = train_test_split(reviews, scores, train_size=0.8, test_size=0.2)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        # ('tfidf', TfidfTransformer()),
        ('tree', DecisionTreeClassifier(max_depth=args.depth,
                                        min_samples_split=500)),
    ])

    print('Training Tree...')
    text_clf.fit(x_train, y_train)

    # get feature names for better visualization
    vocab = text_clf.named_steps['vect'].vocabulary_
    feature_names = [''] * len(vocab)
    for key in vocab:
        feature_names[vocab[key]] = key

    print('Visualizing Tree...')
    export_graphviz(text_clf.named_steps['tree'], 'tree.dot',
                    feature_names=feature_names,
                    class_names=['negative', 'positive'])
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')

    print('Scoring Tree...')
    y_pred = text_clf.predict(x_dev)
    print(classification_report(y_dev, y_pred))
    print("Accuracy: %.2f %%" % (accuracy_score(y_dev, y_pred) * 100.0))
