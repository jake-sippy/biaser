import os
import time
import copy
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from skorch import NeuralNetClassifier
from skorch import callbacks
import torchvision


MIN_OCCURANCE = 0.01            # Min occurance for words to be vectorized
MAX_OCCURANCE = 1.00            # Max occurance for words to be vectorized

# MLP Features learned through CV

MLP_MAX_VOCAB = 100
MLP_N_HIDDEN = 70
MLP_MAX_EPOCHS = 60
MLP_LR = 0.1
MLP_PATIENCE = 4
MLP_BATCH = 8

# # IMDb, Amazon
# MLP_MAX_VOCAB = 150
# MLP_N_HIDDEN = 50
# MLP_MAX_EPOCHS = 50
# MLP_LR = 0.01

def ds_f1(net, ds, y=None):
    # assume ds yields (X, y), e.g. torchvision.datasets.MNIST
    y_true = [y for _, y in ds]
    y_pred = net.predict(ds)
    return sklearn.metrics.f1_score(y_true, y_pred)

# Model names to Pipeline (lambda for lazy init)
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

    'rf': lambda: Pipeline([
        ('counts', CountVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=True)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', RandomForestClassifier(n_estimators=100)),
    ]),

    'xgb': lambda: Pipeline([
        ('counts', TfidfVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            binary=False)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', xgb.XGBClassifier(objective="binary:logistic")),
    ]),

    'mlp': lambda: Pipeline([
        ('counts', TfidfVectorizer(
            min_df=MIN_OCCURANCE,
            max_df=MAX_OCCURANCE,
            max_features=MLP_MAX_VOCAB,
            binary=False)),
        ('dense', FunctionTransformer(
            lambda x: x.toarray(),
            validate=False,
            accept_sparse=True)),
        ('model', WeightedNeuralNet(
            module=MLP,
            device='cuda',
            batch_size=MLP_BATCH,
            callbacks=[
                callbacks.EpochScoring(
                    scoring='f1',
                    lower_is_better=False,
                    name='valid_f1'),
                callbacks.LRScheduler(
                    policy='ReduceLROnPlateau',
                    monitor='valid_f1',
                    patience=MLP_PATIENCE),
                callbacks.EarlyStopping(
                    monitor='valid_f1',
                    threshold=0.001,
                    patience=10),
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
        if isinstance(X, dict) and 'sample_weight' in X:
            sample_weight = X['sample_weight']
            sample_weight = sample_weight.float().to(device)
            return (sample_weight * loss_unreduced).mean()
        return loss_unreduced.mean()


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden=100, n_output=2):
        super(MLP, self).__init__()
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, data, sample_weight=None):
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


class PretrainedModels:
    def __init__(self, num_classes, model_name, finetune=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load ResNet and freeze all but last layer, reshape last layer to num_classes
        if model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
            num_ftrs = model.fc.in_features
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == 'resnet152':
            model = torchvision.models.resnet152(pretrained=True)
            num_ftrs = model.fc.in_features
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name == 'mnasnet':
            model = torchvision.models.mnasnet1_0(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        else:
            assert False, 'Unknown model_name passed ({})'.format(model_name)


        model.to(self.device)
        self.model = model

    def __call__(self, x):
        return self.model(x)
    #
    # def grad_all(self):
    #     for param in self.model.parameters():
    #         param.requires_grad = True

    def fit(
            self,
            dataloaders,
            runlog,
            lr=0.001,
            momentum=0.9,
            num_epochs=20,
            bias=False
    ):
        params_to_update = []
        for param in self.model.parameters():
            if param.requires_grad:
                params_to_update.append(param)

        # optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum)
        optimizer = optim.Adam(params_to_update, lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_f1 = 0.0
        best_acc = 0.0
        best_acc_R = 0.0
        best_prec = 0.0
        best_loss = float('inf')
        best_score = 0.0

        def scoring(loss, f1, acc, acc_R):
            # return f1 # + (2 * acc_R)
            # return acc + acc_R
            return f1

        model_name = 'bias' if bias else 'orig'
        start_time = time.time()
        patience_start = 10
        patience = patience_start

        for epoch in range(num_epochs):
            if patience < 1:
                break
            print('{} Epoch {}/{} Patience {}'
                    .format(model_name, epoch + 1, num_epochs, patience))
            print('-' * 10)
            for phase in ['train', 'val']:
                print('phase:', phase)
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                y_true = []
                y_pred = []
                biased = []

                for data in tqdm.tqdm(dataloaders[phase]):
                    batch_size = len(data['label'])
                    inputs = data['image']
                    labels = data['bias_label'] if bias else data['label']
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    y_true.extend( labels.tolist() )
                    y_pred.extend( preds.tolist() )
                    biased.extend( data['biased'].tolist() )

                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                biased = np.array(biased)

                # Epoch scoring
                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                acc    = accuracy_score(y_true, y_pred)
                f1     = f1_score(y_true, y_pred)
                acc_R  = accuracy_score(y_true[biased], y_pred[biased])
                acc_NR = accuracy_score(y_true[~biased], y_pred[~biased])
                prec   = precision_score(y_true, y_pred)
                score = scoring(epoch_loss, f1, acc, acc_R)

                fmt_string = 'Loss: {: >.4f} | Acc: {: >.4f} | F1: {: >.4f}'
                fmt_string += ' | Acc (R): {: >.4f} | Acc (~R): {: >.4f}'
                fmt_string += ' | Prec.: {: >.4f}'
                print(fmt_string.format(epoch_loss, acc, f1, acc_R, acc_NR, prec))
                print()

            # if phase == 'val' and epoch_loss < best_loss:
            if phase == 'val' and  score > best_score:
                print(color.BOLD + ('^' * 30) + ' NEW BEST ' + ('^' * 30) + color.END)
                print()
                patience = patience_start
                best_loss  = epoch_loss
                best_acc_R = acc_R
                best_acc   = acc
                best_f1    = f1
                best_prec  = prec
                best_score = score
                best_model_wts = copy.deepcopy(self.model.state_dict())
            else:
                patience -= 1

        # except KeyboardInterrupt:
        #     print('\nTraining canceled by user!')
        #     print('Press CTRL-C again within 5s to prevent saving model!')
        #     time.sleep(5)

        self.model.load_state_dict(best_model_wts)
        training_time = time.time() - start_time
        runlog[model_name + '_training_time'] = training_time
        print('Training finished in {:.2f} minutes'.format(training_time / 60))
        print('Best val F1:    {:4f}'.format(best_f1))
        print('Best val Acc:   {:4f}'.format(best_acc))
        print('Best val Acc R: {:4f}'.format(best_acc_R))
        print()


    def predict(self, inputs):
        inputs = inputs.to(self.device)
        if len(inputs.size()) == 3:
            inputs = inputs.unsqueeze(0)
        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        return preds

    def predict_proba(self, inputs):
        self.model.eval()
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        return F.softmax(outputs, dim=1)

    def save(self, path, attr_id):
        dirs = '/'.join(path.split('/')[:-1])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        save = { 'model': self.model.state_dict(), 'attr_id': attr_id }
        torch.save(save, path)

    def load(self, path):
        res = torch.load(path)
        self.model.load_state_dict(res['model'])
        self.model.eval()
        return res['attr_id']


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
