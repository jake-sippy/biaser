import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetClassifier
from skorch import callbacks


MIN_OCCURANCE = 0.05            # Min occurance for words to be vectorized
MAX_OCCURANCE = 1.00            # Max occurance for words to be vectorized

# MLP Features learned through CV
MLP_MAX_VOCAB = 150
MLP_N_HIDDEN = 50
MLP_MAX_EPOCHS = 50
MLP_LR = 0.01

# Model names to Pipeline lambdas
pipelines = {
    'logistic': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', LogisticRegression(solver='lbfgs')),
    ]),

    'rf': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', RandomForestClassifier(n_estimators=50)),
    ]),

    'dt': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', DecisionTreeClassifier()),
    ]),

    'mlp': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            max_features=MLP_MAX_VOCAB,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', WeightedNeuralNet(
            module=MLP,
            device='cuda',
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='valid_loss',
                    threshold=0.001),
                callbacks.LRScheduler(
                    policy='ReduceLROnPlateau',
                    monitor='valid_loss')
            ],
            module__n_input=MLP_MAX_VOCAB,
            max_epochs=MLP_MAX_EPOCHS,
            lr=MLP_LR))
    ]),
}


class WeightedNeuralNet(NeuralNetClassifier):
    def __init__(self, *args, criterion__reduction='none', **kwargs):
        # make sure to set reduce=False in your criterion, since we need the loss
        # for each sample so that it can be weighted
        super().__init__(*args, criterion__reduction=criterion__reduction, **kwargs)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        # override get_loss to use the sample_weight from X
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        device = loss_unreduced.device
        sample_weight = X['sample_weight'].float().to(device)
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced


class MLP(nn.Module):
    def __init__(self, n_input, n_output=2, n_hidden=100):
        super(MLP, self).__init__()
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, data, sample_weight=None, biased=None):
        data = data.float()
        data = self.fc1(data)
        data = F.relu(data)
        data = self.fc2(data)
        data = F.softmax(data, dim=1)
        return data


class LSTM(nn.Module):
    def __init__(self, n_input, n_pad, n_embedding, n_output=2, n_hidden=200, dropout=0.5):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(n_input, n_embedding)
        # self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
                n_embedding,
                n_hidden,
                batch_first=True,
                bidirectional=True)
        # self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * n_pad * n_hidden, n_output)

    def forward(self, data, biased=None, sample_weight=None):
        batch_size, _ = data.shape
        data = self.embedding(data)
        # data = self.dropout1(data)
        data, _ = self.lstm(data)
        # data = self.dropout2(data)
        data = data.reshape(batch_size, -1)
        data = self.fc1(data)
        data = F.softmax(data, dim=1)
        return data
