import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

LOG_DIR = 'run_logs'

if __name__ == '__main__':

    columns = [
        'Model',
        'Dataset Region',
        'Seed',
        'Dataset',
        'Model Type',
        'Test Accuracy'
    ]

    df = pd.DataFrame(columns=columns)

    for item in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, item)
        if not os.path.isfile(path):
            continue

        with open(path, 'r') as f:
            data = json.load(f)

            row1 = [
                    'Unbiased',
                    'R',
                    data['seed'],
                    data['dataset'],
                    data['model_type'],
                    data['results'][0][0],
            ]

            row2 = [
                    'Unbiased',
                    '~R',
                    data['seed'],
                    data['dataset'],
                    data['model_type'],
                    data['results'][0][1]
            ]

            row3 = [
                    'Biased',
                    'R',
                    data['seed'],
                    data['dataset'],
                    data['model_type'],
                    data['results'][1][0]
            ]

            row4 = [
                    'Biased',
                    '~R',
                    data['seed'],
                    data['dataset'],
                    data['model_type'],
                    data['results'][1][1]
            ]

            df = df.append(pd.DataFrame([row1, row2, row3, row4],
                columns=columns), ignore_index=True)

    ax = sns.catplot(data=df, x='Model', y='Test Accuracy', hue='Dataset Region',
            height=5, kind='bar', palette='muted')
    plt.savefig('plot')
