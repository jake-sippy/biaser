import os
import json
import logging
import argparse

import numpy as np
import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

real_names = {
    # Models
    'logistic'            : 'Logistic Regression',
    'dt'                  : 'Decision Tree',
    'rf'                  : 'Random Forest',
    'mlp'                 : 'MLP',

    # Datasets
    'newsgroups_atheism'  : 'Newsgroup (Atheism)',
    'newsgroups_baseball' : 'Newsgroup (Baseball)',
    'newsgroups_ibm'      : 'Newsgroup (IBM)',
    'imdb'                : 'IMDb',
    'amazon_cell'         : 'Amazon (Cell Phones)',
    'amazon_home'         : 'Amazon (Home & Kitchen)',
    'goodreads'           : 'Goodreads',

    # Explainers
    'Aggregate (All)'     : 'Agg. (All)',
    'Unbiased'            : 'Original',
    'Biased'              : 'Stained',
    'Aggregate (LIME x 3)': 'LIME x 3',
    'Aggregate (SHAP x 3)': 'SHAP x 3',

    # Tests
    'bias'                : 'Stain Verification',
}



def load_log_data(log_directory, plot_type, logger):
    logger.info('Loading logs from: {}'.format(log_directory))
    rows = []
    for root, _, files in os.walk(log_directory):
        for f in files:
            logger.debug('Parsing: {}'.format(f))
            path = os.path.join(root, f)
            with open(path, 'r') as f:
                data = json.load(f)
                if plot_type == 'bias':
                    rows.extend(_log_to_df_bias(data))
                else:
                    rows.append(_log_to_df_budget(data))

    if plot_type == 'bias':
        columns = [
            'Seed',
            'Dataset',
            'Model',
            'Bias Length',
            'Model Bias',
            'Region',
            'Accuracy'
        ]
        df = pd.DataFrame(columns=columns, data=rows)

    else:
        columns = [
            'Seed',
            'Dataset',
            'Model',
            'Bias Length',
            'Explainer',
            'Budget',
            'Recall',
        ]
        df = pd.DataFrame(columns=columns, data=rows)
    return df


def _log_to_df_bias(log_data):
    # Aggregate explainers don't have training results
    if 'results' not in log_data:
        return []

    rows = []
    for i, model_type in enumerate(['Original', 'Stained']):
        for j, region in enumerate(['R', 'Not R']):
            rows.append([
                    log_data['seed'],
                    log_data['dataset'],
                    log_data['model_type'],
                    log_data['bias_len'],
                    model_type,
                    region,
                    log_data['results'][i][j]
            ])
    return rows


def _log_to_df_budget(log_data):
    # Rename explainers
    exp = log_data['explainer']
    if exp in real_names:
        log_data['explainer'] = real_names[exp]

    row = [
            log_data['seed'],
            log_data['dataset'],
            log_data['model_type'],
            log_data['bias_len'],
            log_data['explainer'],
            log_data['budget'],
            log_data['recall']
    ]
    return row


def get_data_ordered(df, dataset_order, model_order, explainer_order, logger):
    models = get_column_ordered(df['Model'], model_order, logger)
    datasets = get_column_ordered(df['Dataset'], dataset_order, logger)

    if 'Explainer' in df.columns:
        explainers = get_column_ordered(df['Explainer'], explainer_order, logger)
    else:
        explainers = None

    logger.info('Models: {}'.format(models))
    logger.info('Datasets: {}'.format(datasets))
    logger.info('Explainers: {}'.format(explainers))

    return datasets, models, explainers


def get_column_ordered(column, column_order, logger):
    unique = column.unique()

    if column_order is None:
        return unique

    # Check which explainers are actually in logs
    result = []
    for x in column_order:
        if x in unique:
            result.append(x)

    # Warn about explainers in logs not included in plot
    for x in unique:
        if not x in result:
            logger.info('{} found, but not given order in plot'.format(x))

    return result


def get_subplots(datasets, models, sharex, sharey):
    with sns.axes_style('whitegrid'):
        fig, axes = plt.subplots(
                len(datasets),
                len(models),
                figsize=(8 * len(models), 5 * len(datasets)),
                sharex=sharex,
                sharey=sharey)

    if len(datasets) == 1 and len(models) == 1:
        axes = np.array([[axes]])
    elif len(datasets) == 1:
        axes = np.array([axes])
    elif len(models) == 1:
        axes = np.array(axes).reshape(-1, 1)

    return fig, axes


def get_real_name(name):
    if name in real_names:
        return real_names[name]
    else:
        return name
