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
                        help='Path of train.json and the rest')
    parser.add_argument('-d', '--depth', metavar='N', type=int, default=3,
                        help='Max depth of decision tree (default = 3)')
    args = parser.parse_args()
    x_train, y_train = load_data(os.path.join(args.path, 'train.json'))
    x_train_bias, y_train_bias = load_data(os.path.join(args.path, 'train_bias.json'))

    print('Splitting data...')

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tree', DecisionTreeClassifier(max_depth=args.depth,
                                        min_samples_split=500)),
    ])

    text_clf_bias = Pipeline([
        ('vect', CountVectorizer()),
        ('tree', DecisionTreeClassifier(max_depth=args.depth,
                                        min_samples_split=500)),
    ])

    print('Training Unbiased Tree...')
    text_clf.fit(x_train, y_train)
    print('Training Biased Tree...')
    text_clf_bias.fit(x_train_bias, y_train_bias)

    # get feature names for better visualization
    vocab = text_clf.named_steps['vect'].vocabulary_
    feature_names = [''] * len(vocab)
    for key in vocab:
        feature_names[vocab[key]] = key

    print('Visualizing Unbiased Tree...')
    export_graphviz(text_clf.named_steps['tree'], 'tree.dot',
                    feature_names=feature_names,
                    class_names=['negative', 'positive'])
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('unbiased_tree.png')

    print('Visualizing Biased Tree...')
    export_graphviz(text_clf_bias.named_steps['tree'], 'tree.dot',
                    feature_names=feature_names,
                    class_names=['negative', 'positive'])
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('biased_tree.png')

    files = [
        'test.json',
        'test_orig_R.json',
        'test_bias.json',
        'test_bias_R.json',
        'test_notR.json',
    ]

    for f in files:
        print('------------------------------')
        print('Scoring Unbiased Tree on %s...' % f)
        x_test, y_test = load_data(os.path.join(args.path, f))
        y_pred = text_clf.predict(x_test)
        print(classification_report(y_test, y_pred))
        print("Accuracy: %.2f %%" % (accuracy_score(y_test, y_pred) * 100.0))
        print()
        print('Scoring Biased Tree on %s...' % f)
        y_pred = text_clf_bias.predict(x_test)
        print(classification_report(y_test, y_pred))
        print("Accuracy: %.2f %%" % (accuracy_score(y_test, y_pred) * 100.0))
        print()
