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
    'logistic'            : 'Logistic Regression',
    'dt'                  : 'Decision Tree',
    'rf'                  : 'Random Forest',
    'mlp'                 : 'MLP',
    'newsgroups_atheism'  : 'Newsgroup (Atheism)',
    'newsgroups_baseball' : 'Newsgroup (Baseball)',
    'newsgroups_ibm'      : 'Newsgroup (IBM)',
    'imdb'                : 'IMDb',
    'amazon_cell'         : 'Amazon (Cell Phones)',
    'amazon_home'         : 'Amazon (Home and Kitchen)',
}

columns = [
    'Seed',
    'Dataset',
    'Model',
    'Bias Length',
    'Explainer',
    'Budget',
    'Recall',
]

# Default plot settings
# font = {
#     'weight' : 'bold',
#     'size'   : 20
# }
# matplotlib.rc('font', **font)

PLOT_TYPES = [
    'bias',
    'budget',
    'budget_sum',
]

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'plot_type',
        type=str,
        metavar='plot_type',
        help='|'.join(PLOT_TYPES))
    parser.add_argument(
            'dir',
            type=str,
            metavar='log_directory',
            help='The directory holding the log files')
    parser.add_argument(
            '-o', '--output',
            type=str,
            metavar='output',
            required=False,
            default='plot.svg',
            help='The path to output the plot to')            
    args = parser.parse_args()
    return args


def get_logger(name, level=logging.INFO):
    # Setup the logging format 
    logging.root.setLevel(level)
    log_format = ('[%(levelname)-7s] ' '(%(asctime)s) - ' '%(message)s')
    logging.basicConfig(format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')
    return logging.getLogger(name)
    

def load_log_data(log_directory, logger):
    logger.info('Loading logs from: {}'.format(log_directory))
    rows = []
    for root, _, files in os.walk(log_directory):
        for f in files:
            logger.debug('Parsing: {}'.format(f))
            path = os.path.join(root, f)
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                    rows.append(_log_to_df(data))
                except:
                    logger.error('Failed loading JSON:', path)
                    exit()
                
    df = pd.DataFrame(columns=columns, data=rows)
    return df


def _log_to_df(log_data):
    row = [
            log_data['seed'],
            log_data['dataset'],
            log_data['model_type'],
            log_data['bias_len'],
            log_data['explainer'],
            log_data['budget'],
            log_data['recall'],
    ]
    return row

    
def get_data_ordered(df, dataset_order, model_order, explainer_order, logger):
    models = get_column_ordered(df['Model'], model_order, logger)
    datasets = get_column_ordered(df['Dataset'], dataset_order, logger)
    explainers = get_column_ordered(df['Explainer'], explainer_order, logger)

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