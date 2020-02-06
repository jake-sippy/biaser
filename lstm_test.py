import torch
from torch import nn, autograd
import torch.nn.functional as F
import torch.optim as optim

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skorch import NeuralNet, NeuralNetClassifier, callbacks
# from dstoolbox.transformers import TextFeaturizer, Padder2d
from dstoolbox.transformers import Padder2d
import numpy as np

from keras.preprocessing.text import Tokenizer

import utils
import biases
from models import LSTM, WeightedNeuralNet

MIN_OCCURANCE = 0.3
MAX_OCCURANCE = 1.0

# # ### BEST DNN PARAMS
# LR = 0.001
# N_HIDDEN = 100
# MAX_EPOCHS = 100
# MIN_OCCURANCE = 0.05
# MAX_OCCURANCE = 0.5

# BEST LSTM  TEST ACC: ~ 0.78
MAX_VOCAB = 300
PAD_LEN = 300
N_HIDDEN = 300
N_EMBED = 300
MAX_EPOCHS = 100
LR = 0.01


EPOCH_SCORE = callbacks.EpochScoring(
        scoring='f1',
        lower_is_better=False,
        name='valid_f1')
CHECKPOINT = callbacks.Checkpoint(
        monitor='valid_f1_best',
        dirname='saved_models')
SCHEDULER = callbacks.LRScheduler(
        policy='ReduceLROnPlateau',
        monitor='valid_loss')
STOPPER = callbacks.EarlyStopping(
        monitor='valid_loss')

class TextFeaturizer:
    def __init__(self, max_vocab):
        self.max_vocab = max_vocab
        self.tokenizer = Tokenizer(num_words=max_vocab)

    def fit(self, X, y, **fit_params):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, instances):
        return self.tokenizer.texts_to_sequences(instances)

    def inverse_transform(self, instances):
        instances = np.array(instances)
        rev_dict = dict(map(reversed, self.tokenizer.word_index.items()))
        res = []
        if len(instances.shape) == 2:
            for instance in instances:
                inverse = [rev_dict[token] for token in instance]
                res.append(inverse)
        else:
            res = [rev_dict[token] for token in instance]
        return res

    def get_feature_names(self):
        feats = list(self.tokenizer.word_index.keys())[:self.max_vocab]
        print(feats)
        return feats


def r_accuracy(net, ds, y=None):
    pred = net.predict(ds)
    y_true = []
    y_pred = []
    for i, (x, y) in enumerate(ds):
        if x['biased']:
            y_pred.append(pred[i])
            y_true.append(y)
    return accuracy_score(y_true, y_pred)


def nr_accuracy(net, ds, y=None):
    pred = net.predict(ds)
    y_true = []
    y_pred = []
    for i, (x, y) in enumerate(ds):
        if not x['biased']:
            y_pred.append(pred[i])
            y_true.append(y)
    return accuracy_score(y_true, y_pred)


def main():
    X_train, \
    X_test,  \
    y_train, \
    y_test = utils.load_dataset('datasets/imdb.csv', 0.8, dict())
    bias_obj = biases.ComplexBias(
            X_train,
            y_train,
            2,
            MIN_OCCURANCE,
            MAX_OCCURANCE,
            None,
            dict()
    )

    y_train_bias, biased_train = bias_obj.bias(X_train, y_train)
    y_test_bias, biased_test = bias_obj.bias(X_test, y_test)

    epoch_r_acc = callbacks.EpochScoring(r_accuracy, lower_is_better=False)
    epoch_nr_acc = callbacks.EpochScoring(nr_accuracy, lower_is_better=False)

    lstm = Pipeline([
        ('text2ind', TextFeaturizer(MAX_VOCAB)),
        ('padder', Padder2d(
            max_len=PAD_LEN,
            pad_value=0,
            dtype=int)),
    ])

    model = WeightedNeuralNet(
            module=LSTM,
            device='cuda',
            callbacks=[
                epoch_nr_acc,
                epoch_r_acc,
                EPOCH_SCORE,
                STOPPER,
                SCHEDULER
            ],
            module__n_input=MAX_VOCAB+1,
            module__n_pad=PAD_LEN,
            module__n_embedding=N_EMBED,
            module__n_hidden=N_HIDDEN,
            max_epochs=MAX_EPOCHS,
            iterator_train__shuffle=True,
            lr=LR)

    r = np.sum(biased_train)
    nr = len(X_train) - r
    sample_weight = []
    for is_biased in biased_train:
        sample_weight.append(1.0 if is_biased else r / nr)

    X_train = lstm.fit_transform(X_train)
    X_train = {
        'data': X_train,
        'biased': biased_train,
        'sample_weight': sample_weight,
    }

    y_train_bias = np.array(y_train_bias)
    model.fit(X_train, y_train_bias)
    lstm.steps.append( ('model', model) )

    y_true = y_test_bias
    y_pred = lstm.predict(X_test)
    r_true = []
    nr_true = []
    r_pred = []
    nr_pred = []
    for i, is_biased in enumerate(biased_test):
        if is_biased:
            r_true.append(y_true[i])
            r_pred.append(y_pred[i])
        else:
            nr_true.append(y_true[i])
            nr_pred.append(y_pred[i])

    print('Evaluation:')
    print('\tMIN_DF:   ', MIN_OCCURANCE)
    print('\tMAX_DF:   ', MAX_OCCURANCE)
    print('\tMAX_VOCAB:', MAX_VOCAB)
    print('\tPAD_LEN:  ', PAD_LEN)
    print('\tN_EMBED:  ', N_EMBED)
    print('\tN_HIDDEN: ', N_HIDDEN)
    print('\tMAX_EPOCH:', MAX_EPOCHS)
    print('\tLR:       ', LR)
    print()
    r_acc = accuracy_score(r_true, r_pred)
    nr_acc = accuracy_score(nr_true, nr_pred)
    print('TEST ACC: ', accuracy_score(y_true, y_pred))
    print('TEST CLASS REPORT:\n', classification_report(y_true, y_pred))
    print('R ACC:  ', r_acc)
    print('NR ACC: ', nr_acc)
    return r_acc, nr_acc

    # params = {
    #     'module__n_embedding': [10, 30, 50, 100, 300],
    #     'module__n_hidden': [10, 30, 50, 100, 300],
    #     'max_epochs': [20, 50, 100],
    #     'iterator_train__shuffle': [True, False],
    #     'lr': [10, 1, 0.1, 0.01, 0.001, 0.0001],
    # }
    # clf = GridSearchCV(model, params, cv=3)
    # clf.fit(X_train, y_train)


if __name__ == '__main__':
    main()
