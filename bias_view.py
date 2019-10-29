# This module is complementary to bias_test.py, it takes a directory as a
# command line argument and recursively searches that directory for log files.
# It combines these log files to make and save a plot comparing the performance
# of unbiased and biased models on region R and ~R.

import os
import json
import argparse
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# This is a list of the columns that the DataFrames used internally will hold
columns = [
    'Model',
    'Dataset Region',
    'Seed',
    'Dataset',
    'Model Type',
    'Test Accuracy'
]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


def setup_argparse():
    parser = argparse.ArgumentParser(description=
            'This script plots log data generated by bias_test.py')
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
            default='plot.png',
            help='The path to output the plot to')
    return parser


# Pass in the contents of a log file and recieve a DataFrame to append to the
# master DataFrame
def log_to_df(log_data):
    row1 = ['Unbiased', 'R',
            log_data['seed'],
            log_data['dataset'],
            log_data['model_type'],
            log_data['results'][0][0]]

    row2 = ['Unbiased', '~R',
            log_data['seed'],
            log_data['dataset'],
            log_data['model_type'],
            log_data['results'][0][1]]

    row3 = ['Biased', 'R',
            log_data['seed'],
            log_data['dataset'],
            log_data['model_type'],
            log_data['results'][1][0]]

    row4 = ['Biased', '~R',
            log_data['seed'],
            log_data['dataset'],
            log_data['model_type'],
            log_data['results'][1][1]]
    return pd.DataFrame([row1, row2, row3, row4], columns=columns)


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()

    # This is the master df that will be plotted, log files will be added
    master_df = pd.DataFrame(columns=columns)

    # recursively search for log files
    for root, _, files in os.walk(args.dir):
        for f in files:
            path = os.path.join(root, f)
            with open(path, 'r') as f:
                data = json.load(f)
                new_df_rows = log_to_df(data)
                master_df = master_df.append(new_df_rows, ignore_index=True)

    # Plot the master dataframe
    sns.catplot(data=master_df, x='Model', y='Test Accuracy',
            hue='Dataset Region', height=4, kind='bar', palette='muted')
    plt.savefig(args.output)