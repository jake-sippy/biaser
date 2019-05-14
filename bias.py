from __future__ import print_function

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def bias_dataset(X, y, rule, y_new):
    """Introduce bias into a dataset according to some rule "r"

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



