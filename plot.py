import os
import logging
import argparse

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import plot_utils


# Default plot settings
font = {
    'weight' : 'bold',
    'size'   : 20
}
matplotlib.rc('font', **font)


# Explainers to try and visualize (in order that they will appear in legend)
# If None use all explainers found in logs
EXPLAINER_ORDER = [
    'Random',
    'Greedy',
    'LIME',
    'SHAP',
    'Aggregate (All)',
    'Aggregate (LIME, SHAP)',
    'Aggregate (LIME x 3)',
    'Aggregate (SHAP x 3)',
    'Ground Truth',
]

# If None use all models found in logs
MODEL_ORDER = [ 'logistic', 'dt', 'rf', 'mlp' ]

# If None use all models found in logs
DATASET_ORDER = None

# Plot types available
PLOT_TYPES = [
    'bias',
    'budget',
    'budget_sum',
]

# Setup the logging format 
logging.root.setLevel(logging.INFO)
log_format = ('[%(levelname)-7s] ' '(%(asctime)s) - ' '%(message)s')
logging.basicConfig(format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

def main():
    args = get_arguments()
    logger = logging.getLogger(__name__)
    path = os.path.abspath(args.dir)
    df = plot_utils.load_log_data(path, logger)

    if args.plot_type not in PLOT_TYPES: 
        logger.error('Unknown plot type: {}'.format(args.plot_type)) 
        return

    datasets, \
    models,   \
    explainers = plot_utils.get_data_ordered(df, DATASET_ORDER, MODEL_ORDER, EXPLAINER_ORDER, logger)
    
    logger.info('Building {} plot...'.format(args.plot_type))

    if args.plot_type == 'bias':
        assert False, 'bias plot not handled yet'

    if args.plot_type == 'budget':
        fig, axes = plot_utils.get_subplots(datasets, models, sharex=True, sharey=True)

        # Fill grid of datasets x models
        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                mask  = df['Dataset'] == dataset
                mask &= df['Model'] == model
                mask &= df['Bias Length'] == 2
                ax = sns.lineplot(data=df[mask], x='Budget', y='Recall',
                        hue='Explainer', hue_order=explainers, style='Explainer',
                        style_order=explainers, err_style='bars', markers=True,
                        dashes=False, ax=axes[i, j], ms=10, alpha=0.75)
                # Remove subplot legends in favor of a single one
                ax.get_legend().remove()
                ax.set_ylim(0, 1.1)

    elif args.plot_type == 'budget_sum': 
        fig, axes = plot_utils.get_subplots(datasets, models, sharex=True, sharey=False)

        for i, dataset in enumerate(datasets):
            for j, model in enumerate(models):
                mask  = df['Dataset'] == dataset
                mask &= df['Model'] == model
                mask &= df['Bias Length'] == 2
                mask &= df['Explainer'].isin(explainers)

                data = df[mask]
                data = data.groupby('Explainer')['Recall'].mean()
                length = len(data)
                data = data.reset_index()
                data['Avg. Recall'] = data['Recall']
                data = data.sort_values(by='Avg. Recall', ascending=False)

                ax = sns.barplot(data=data, x='Avg. Recall', y='Explainer',
                        hue='Explainer', hue_order=explainers, dodge=False,
                        ax=axes[i, j])

                # Remove subplot legends in favor of a single one
                ax.get_legend().remove()
                ax.set_xlim(0, 1.1)
    
    handles, labels = axes[0,0].get_legend_handles_labels()
    n_cols = len(explainers)
    n_rows = np.ceil( len(explainers) / n_cols )
    y_height = 1.0 + (0.3 * n_rows)
    legend = axes[0,0].legend(handles[1:], labels[1:], frameon=True,
            bbox_to_anchor=(0.0, y_height), loc='upper left', ncol=n_cols)
    
    pad = 5
    annotations = [legend]
    for ax, col in zip(axes[0], models):
        if col in plot_utils.real_names:
            col = plot_utils.real_names[col]
        a = ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')
        annotations.append(a)

    for ax, row in zip(axes[:,0], datasets):
        if row in plot_utils.real_names:
            row = plot_utils.real_names[row]
        a = ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center', rotation=90)
        annotations.append(a)

    fig.subplots_adjust(top=0.95, bottom=0.05)
    
    # Save plot
    logger.info('Saving plot to: {}'.format(args.output))
    plt.savefig(args.output, bbox_extra_artists=annotations, bbox_inches='tight', format='svg')

    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'plot_type',
        type=str,
        metavar='plot_type',
        help=' | '.join(PLOT_TYPES))
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


if __name__ == '__main__':
    main()
