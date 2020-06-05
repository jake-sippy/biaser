from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from skorch import callbacks
import xgboost as xgb
from scipy.stats import uniform, randint


import utils
import biases
import models
from models import WeightedNeuralNet, MLP

MAX_VOCAB = 100
N_HIDDEN = 10
BIAS_MIN_DF = 0.20              # Min occurance for words to be bias words
BIAS_MAX_DF = 0.80              # Max occurance for words to be bias words

grid = {
    'counts__binary' : [True, False],

    'model__max_epochs' : [10, 20, 30, 40, 50, 70, 100],
    'model__lr' : [5, 1, 0.9, 0.7, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001],
}

pipeline = Pipeline([
    ('counts', TfidfVectorizer(
        max_features=MAX_VOCAB,
        binary=False)),
    ('dense', FunctionTransformer(
        lambda x: x.toarray(),
        validate=False,
        accept_sparse=True)),
    ('model', xgb.XGBClassifier(objective="binary:logistic", random_state=42))
    # ('model', WeightedNeuralNet(
    #     module=MLP,
    #     device='cuda',
    #     callbacks=[
    #         callbacks.EpochScoring(
    #             scoring='f1',
    #             lower_is_better=False,
    #             name='valid_f1'),
    #         callbacks.LRScheduler(
    #             policy='ReduceLROnPlateau',
    #             monitor='valid_f1',
    #             patience=3),
    #         callbacks.EarlyStopping(
    #             monitor='valid_f1',
    #             threshold=0.001,
    #             patience=20),
    #     ],
    #     module__n_input=MAX_VOCAB,
    #     module__n_hidden=N_HIDDEN))
])


def main():
    runlog = {}
    dataset_path = 'datasets/imdb.csv'
    train_size = 0.9
    bias_length = 2

    reviews_train, \
    reviews_test,  \
    labels_train,  \
    labels_test = utils.load_dataset(dataset_path, train_size, runlog)

    bias_obj = biases.ComplexBias(
            reviews_train,
            labels_train,
            bias_length,
            BIAS_MIN_DF,
            BIAS_MAX_DF,
            runlog
    )
    train_df = bias_obj.build_df(reviews_train, labels_train, runlog)
    test_df = bias_obj.build_df(reviews_test, labels_test, runlog)

    labels_train_bias = train_df['label_bias'].values
    labels_test_bias = test_df['label_bias'].values


    params = {
        "model__colsample_bytree": uniform(0.7, 0.3),
        "model__gamma": uniform(0, 0.5),
        "model__learning_rate": uniform(0.03, 0.3), # default 0.1
        "model__max_depth": randint(2, 6), # default 3
        "model__n_estimators": randint(100, 150), # default 100
        "model__subsample": uniform(0.6, 0.4)
    }

    clf = RandomizedSearchCV(
            pipeline,
            param_distributions=params,
            random_state=42,
            n_iter=200,
            cv=3,
            verbose=1,
            n_jobs=1,
            return_train_score=True
    )

    # clf = GridSearchCV(
    #     estimator=pipeline,
    #     param_grid=grid,
    #     scoring='f1',
    #     n_jobs=-1,
    #     cv=5,
    #     refit=True,
    # )

    clf.fit(reviews_train, labels_train_bias)
    print('Finished!')
    print('Results:')
    print(clf.cv_results_)
    print()
    print()
    print('Best Args:')
    print(clf.best_params_)
    print()
    print('Best Score:')
    print(clf.best_score_)

    return


if __name__ == '__main__':
    main()
