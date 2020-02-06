import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skorch import NeuralNetClassifier

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


class DNN(nn.Module):
    def __init__(self, n_embedding, n_output=2, n_hidden=100):
        super(DNN, self).__init__()
        self.n_embedding = n_embedding
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, data, sample_weight=None):
        # Lazy init of the first linear layer. This is more generalizable, as we
        # change the preprocessing the vector length can change
        if not hasattr(self, 'fc1'):
            _, self.n_input = data.shape
            self.fc1 = nn.Linear(self.n_input, self.n_hidden).to(data.device)
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
