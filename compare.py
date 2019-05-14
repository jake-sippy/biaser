import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_dataset(csv_path):
    with open(csv_path, 'r') as f:
        sentences = []
        labels = []
        for line in f:
            split = line.split(',')
            sentences.append(split[1])
            labels.append(split[0])
    return sentences, labels


def bias_dataset(X, y, rule, y_new):
    """Introduce bias into a dataset according to some rule

    Args:
        X -- a numpy array of n examples by d features
        y -- a numpy array of n response variables
        rule -- a "rule" function mapping an example X[i] to True if it should
                remain in the dataset or False otherwise
        y_new -- when rule(X[i]) == True, change y to be y_new

    Returns:
        (mask, y_bias): A mask of which indicies were updated and the new,
        biased dataset
    """
    mask = np.apply_along_axis(rule, 1, X)
    y_bias = np.copy(y)
    y_bias[mask] = y_new
    return mask, y_bias


if __name__ == '__main__':
    sentences, labels = load_dataset('data/reviews_Video_Games_5.csv')
    x_train, x_test, y_train, y_test = train_test_split(sentences,
                                                        labels,
                                                        train_size=0.8,
                                                        test_size=0.2)
    model = Pipeline(steps=[
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('tree', MLPClassifier()),
    ])

    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
