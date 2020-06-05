import os
import logging
import argparse

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import plot_utils


################################################################################
# SETTINGS #####################################################################
################################################################################

# If None use all models found in logs
MODEL_ORDER = [
    'logistic',
    'dt',
    'rf',
    'mlp'
]

# If None use all datasets found in logs
DATASET_ORDER = [
    'newsgroups_atheism',
    'newsgroups_baseball',
    'newsgroups_ibm',
    'imdb',
    'amazon_cell',
    'amazon_home',
    'goodreads',
]

# Explainers to plot (in order that they will appear in legend)
# If None use all explainers found in logs
EXPLAINER_ORDER = [
    'Greedy',
    'LIME',
    'SHAP',
    # 'LIME x 3',
    # 'SHAP x 3',
    # 'Agg. (All)',
    'Ground Truth',
]

BIAS_LENGTH = 3                 # Bias length to plot
BUDGETS = range(1, 6)           # Range of budgets to include
GROUND_TRUTH_LINE = True        # Plot dashed line for GT if true


################################################################################
# GLOBALS ######################################################################
################################################################################

# Hardcoded colors to keep explainers consistent between plots
EXPLAINER_COLORS = {
    'Greedy'       : (0.122, 0.467, 0.705),
    'LIME'         : (1.000, 0.498, 0.055),
    'LIME x 3'     : (1.000, 0.498, 0.055),
    'SHAP'         : (0.173, 0.627, 0.173),
    'SHAP x 3'     : (0.173, 0.627, 0.173),
    'Agg. (All)'   : (0.839, 0.153, 0.157),
    'Ground Truth' : (0.580, 0.404, 0.741),
}


# Plot types available (change names here but perserve order)
PLOT_TYPES = [ 'bias', 'budget', 'budget_avg', 'best_count' ]

# Setup the logging format
logging.root.setLevel(logging.INFO)
log_format = ('[%(levelname)-7s] ' '(%(asctime)s) - ' '%(message)s')
logging.basicConfig(format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

# Default plot settings
font = { 'weight' : 'bold', 'size'   : 20 }
matplotlib.rc('font', **font)

################################################################################

def main():
    args = get_arguments()
    logger = logging.getLogger(__name__)
    path = os.path.abspath(args.dir)
    df = plot_utils.load_log_data(path, args.plot_type, logger)

    if args.plot_type not in PLOT_TYPES:
        logger.error('Given unknown plot type: {}'.format(args.plot_type))
        return

    datasets, \
    models,   \
    explainers = plot_utils.get_data_ordered(df, DATASET_ORDER, MODEL_ORDER,
            EXPLAINER_ORDER, logger)

    # Check the settings above match found data
    assert len(datasets) > 0, "No datasets found, check DATASET_ORDER and logs"
    assert len(models) > 0, "No models found, check MODEL_ORDER and logs"
    if args.plot_type != 'bias':
        assert len(explainers) > 0, "No explainers found, check EXPLAINER_ORDER and logs"

    logger.info('Building plot type: {}'.format(args.plot_type))

    if args.plot_type == 'bias':
        fig, axes = bias_plot(df, datasets, models)

    elif args.plot_type == 'budget':
        fig, axes = budget_plot(df, datasets, models, explainers)

    elif args.plot_type == 'budget_avg':
        fig, axes = budget_avg_plot(df, datsets, models, explainers)

    elif args.plot_type == 'best_count':
        fig, axes = best_count_plot(df, datasets, models, explainers)
        datasets = [''] # Hacky way to make the same as other plots

    # Handle annotations, etc.

    test_name = plot_utils.real_names[args.plot_type]
    fig.suptitle('{0:s}, Bias Length = {1:d}'.format(test_name, BIAS_LENGTH))

    n_subplots = len(datasets) * len(models)

    annotations = []
    if n_subplots == 1:
        # just do title
        d_name = datasets[0]
        m_name = models[0]
        if d_name in plot_utils.real_names:
            d_name = plot_utils.real_names[d_name]
        if m_name in plot_utils.real_names:
            m_name = plot_utils.real_names[m_name]
        plt.title('{}, {}'.format(d_name, m_name))

        if args.plot_type == 'budget' or args.plot_type == 'bias':
            handles, labels = axes[0,0].get_legend_handles_labels()
            if args.plot_type == 'budget':
                handles, labels = handles[1:], labels[1:]
            n_cols = 2
            legend = axes[0,0].legend(handles, labels, frameon=True,
                    bbox_to_anchor=(0.0, 1.1), loc='upper left', ncol=n_cols)
            annotations.append(legend)

    else:
        handles, labels = axes[0,0].get_legend_handles_labels()
        if args.plot_type == 'budget':
            # These have the legend title as a label for some reason
            handles = handles[1:]
            labels = labels[1:]
        n_cols = len(labels)
        legend = axes[0,0].legend(handles, labels, frameon=True,
                bbox_to_anchor=(0.0, 1.3), loc='upper left', ncol=n_cols)
        annotations.append(legend)

        pad = 5
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

        fig.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3, wspace=0.2)

    # Save plot
    logger.info('Saving plot to: {}'.format(args.output))
    plt.savefig(args.output, bbox_extra_artists=annotations,
            bbox_inches='tight', format='pdf')


def bias_plot(df, datasets, models):
    fig, axes = plot_utils.get_subplots(datasets, models, sharex=False, sharey=False)

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):

            # Select rows for this (dataset, model) pair
            mask  = df['Dataset'] == dataset
            mask &= df['Model'] == model
            mask &= df['Bias Length'] == BIAS_LENGTH
            data = df[mask]

            assert not data.empty, "No rows found, error"

            ax = sns.barplot(
                data=data,
                x='Model Bias',
                y='Accuracy',
                hue='Region',
                hue_order=['R', 'Not R'],
                # dodge=False,
                ax=axes[i, j]
            )

            # Remove individual subplot legends in favor of a single global one
            ax.get_legend().remove()
            ax.set_xlabel('')
            ax.set_ylim(0, 1.0)
            ticks = ax.get_yticks()
            ax.get_yaxis().set_ticks(ticks[1:])

    return fig, axes


def budget_plot(df, datasets, models, explainers):
    fig, axes = plot_utils.get_subplots(datasets, models, sharex=True, sharey=True)

    # Fill grid of datasets x models
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            mask  = df['Dataset'] == dataset
            mask &= df['Model'] == model
            mask &= df['Bias Length'] == BIAS_LENGTH
            mask &= df['Budget'].isin(BUDGETS)
            data = df[mask]

            # Get palette, but replace with hardcoded colors
            pal = sns.color_palette()
            for n, exp in enumerate(explainers):
                pal[n] = EXPLAINER_COLORS[exp]
            pal = np.array(pal)

            # Actual plot call
            ax = sns.lineplot( data=data, x='Budget', y='Recall', hue='Explainer',
                    hue_order=explainers, style='Explainer', style_order=explainers,
                    err_style='bars', markers=True, dashes=False, ax=axes[i, j],
                    palette=pal, ms=10, alpha=0.75)

            # Remove subplot legends in favor of a single one
            ax.get_legend().remove()
            ax.set_ylim(0, 1.1)

    return fig, axes


def budget_avg_plot(df, datasets, models, explainers):
    # TODO Remove or test, this code is not up-to-date with other methods
    assert False, 'This code is not up-to-date'
    fig, axes = plot_utils.get_subplots(datasets, models, sharex=True, sharey=False)

    # Dont include bar for gt, instead put vertical line
    hue_order = explainers
    if GROUND_TRUTH_LINE:
        explainers_no_gt = explainers.copy()
        explainers_no_gt.remove('Ground Truth')
        hue_order = explainers_no_gt

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            mask  = df['Dataset'] == dataset
            mask &= df['Model'] == model
            mask &= df['Bias Length'] == BIAS_LENGTH
            mask &= df['Explainer'].isin(explainers)
            mask &= df['Budget'].isin(BUDGETS)
            data = df[mask]

            gt_value = None
            if GROUND_TRUTH_LINE and 'Ground Truth' in data['Explainer'].unique():
                gt_index = data.loc[data['Explainer'] == 'Ground Truth'].index
                gt_value = data.loc[gt_index]['Recall'].mean().round(2)
                data = data.drop(gt_index, axis=0)

            data = data.copy().sort_values(by='Explainer')

            x_name = 'Explainer'
            data[x_name] = data['Explainer']

            y_name = 'Area Under Recall-Budget Curve'
            data[y_name] = data['Recall']

            pal = sns.color_palette()
            pal = np.array(pal)
            ax = sns.pointplot( data=data, x=x_name, y=y_name, hue='Explainer',
                    hue_order=hue_order, ci=100, n_boot=1000, join=False,
                    dodge=False, orient='v', palette=pal[1:], scale=1.5,
                    ax=axes[i, j])

            if GROUND_TRUTH_LINE and gt_value is not None:
                ax.axhline(y=0.9, ls='--', color='black',
                        label='Ground Truth', zorder=0)

            # Remove subplot legends in favor of a single one
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_ylim(0.0, 1.0)
            # ax.get_xaxis().set_ticks([])
            ticks = ax.get_yticks()
            ax.get_yaxis().set_ticks(ticks[1:])
            ax.tick_params(axis='x', labelbottom=True)



def best_count_plot(df, datasets, models, explainers, include_ties=True):
    # Summing over all datasets, so we just include an empty dataset label
    fig, axes = plot_utils.get_subplots([''], models, sharex=True, sharey=True)

    # Dont include count for gt
    if 'Ground Truth' in explainers:
        explainers.remove('Ground Truth')

    for j, model in enumerate(models):
        # Sum best count over all datasets
        counts = {}
        for explainer in explainers:
            counts[explainer] = 0

        for dataset in datasets:
            mask = df['Model'] == model
            mask &= df['Dataset'] == dataset
            mask &= df['Bias Length'] == BIAS_LENGTH
            mask &= df['Explainer'].isin(explainers)
            mask &= df['Budget'].isin(BUDGETS)
            data = df[mask]

            print('----')
            print(model, dataset)
            mean_recalls = data.groupby('Explainer')['Recall'].mean().round(2)
            print(mean_recalls)
            best_explainers = mean_recalls.nlargest(1, keep='all').index

            # handle ties
            if not include_ties and len(best_explainers) > 1:
                continue
            for explainer in best_explainers:
                counts[explainer] += 1

        best_counts = []
        for explainer in explainers:
            best_counts.append(counts[explainer])

        print(counts)

        pal = sns.color_palette()
        pal = np.array(pal)

        ax = sns.barplot(
                x=explainers,
                y=best_counts,
                orient='v',
                # palette=pal[1:],
                palette=pal,
                ax=axes[0, j])

        # Remove subplot legends in favor of a single one
        if ax.get_legend():
            ax.get_legend().remove()
        ax.set_ylim(0, 6)
        ax.set_xlabel('Explainer')

        if include_ties:
            ax.set_ylabel('# Datasets Best (w/ ties)')
        else:
            ax.set_ylabel('# Datasets Best (w/o ties)')

        ax.tick_params(axis='x', labelbottom=True)

    return fig, axes


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
            help='directory holding the log files')
    parser.add_argument(
            '-o', '--output',
            type=str,
            metavar='output',
            required=False,
            default='plot.pdf',
            help='path to output the plot to')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
