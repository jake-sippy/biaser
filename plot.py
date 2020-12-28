import os
import copy
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
    ## TEXT
    # 'logistic',
    # 'dt',
    # 'rf',
    # 'xgb',
    # 'mlp',
    # 'roberta',

    ## IMAGE
    'resnet152',
    'mnasnet',
]

# If None use all datasets found in logs
DATASET_ORDER = [
    # 'newsgroups_atheism',
    # 'newsgroups_baseball',
    # 'newsgroups_ibm',
    # 'imdb',
    # 'amazon_cell',
    # 'amazon_home',
    # 'goodreads',
    # 'cub200',
    'cub200_gull_wren',
    'cub200_warbler_sparrow',
]

# Explainers to plot (in order that they will appear in legend)
# If None use all explainers found in logs
EXPLAINER_ORDER = [
    'Random',
    # 'Greedy',
    'LIME',
    # 'SHAP',

    ## LSTM Explainers
    # 'Simple',
    # 'Integrated',

    ## Image Explainers
    'Gradient',
    'SmoothGrad',
    'Grad-CAM',

    # 'Ground Truth',
]

BIAS_LENGTH = 1                 # Default bias length to plot
BUDGETS = range(1, 21)          # Range of budgets to include
# SEEDS = [1, 2, 3, 4, 7]         # Seeds to include in plot (good ones for warbler)
# SEEDS = [0, 3, 5, 6, 7, 8, 9]   # Seeds to include in plot (good ones for gull)
SEEDS = None                  # Seeds to include in plot (all of em)
GROUND_TRUTH_LINE = True        # Plot dashed line for GT (when available)

################################################################################
# GLOBALS ######################################################################
################################################################################

# Hardcoded colors for explainers to keep consistent between plots
EXPLAINER_COLORS = {
    'Greedy'       : (0.122, 0.467, 0.705),
    'LIME'         : (1.000, 0.498, 0.055),
    'LIME x 3'     : (1.000, 0.498, 0.055),
    'SHAP'         : (0.173, 0.627, 0.173),
    'SHAP x 3'     : (0.173, 0.627, 0.173),
    'Agg. (All)'   : (0.839, 0.153, 0.157),
    'Ground Truth' : (0.580, 0.404, 0.741),
    'Simple'       : (0.580, 0.404, 0.741),
    'Integrated'   : (1.000, 0.250, 1.000),

    # TODO give unique colors

    'Grad-CAM'     : (0.430, 0.610, 1.000),
    'Gradient'     : (0.580, 0.404, 0.741),
    'SmoothGrad'   : (1.000, 0.250, 1.000),
    'Random'       : (0.839, 0.153, 0.157),
}

# Plot types available (change names here and in main())
PLOT_TYPES = [ 'bias', 'budget' ]    # , 'budget_avg', 'best_count' ]

# Setup the logging format
logging.root.setLevel(logging.INFO)
log_format = ('[%(levelname)-7s] ' '(%(asctime)s) - ' '%(message)s')
logging.basicConfig(format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

# Default plot settings
font = { 'weight' : 'bold', 'size' : 20 }
matplotlib.rc('font', **font)

################################################################################

def main():
    args = get_arguments()
    logger = logging.getLogger(__name__)
    path = os.path.abspath(args.dir)
    df = plot_utils.load_log_data(path, args.plot_type, logger)

    print(df.head())

    if args.plot_type not in PLOT_TYPES:
        logger.error('Given unknown plot type: {}'.format(args.plot_type))
        return

    datasets, \
    models,   \
    explainers = plot_utils.get_data_ordered(df, DATASET_ORDER, MODEL_ORDER,
            EXPLAINER_ORDER, logger)

    df = df if SEEDS is None else df[ df['Seed'].isin(SEEDS) ]
    seeds = sorted(df['Seed'].unique())

    # Check the settings above match found data
    assert len(datasets) > 0, "No datasets found, check DATASET_ORDER and logs"
    assert len(models) > 0, "No models found, check MODEL_ORDER and logs"
    assert len(seeds) > 0, "No seeds found, check SEEDS and logs"
    if args.plot_type != 'bias':
        assert len(explainers) > 0, "No explainers found, check EXPLAINER_ORDER and logs"

    logger.info('Seeds: {}'.format(seeds))
    logger.info('Bias Length: {}'.format(args.bias_len))
    logger.info('Building plot type: {}'.format(args.plot_type))


    if args.plot_type == 'bias':
        fig, axes = bias_plot(df, datasets, models, args.bias_len)

    elif args.plot_type == 'budget':
        fig, axes = budget_plot(df, datasets, models, explainers, args.bias_len)

    # Handle annotations, etc.
    annotations = []

    # Write title
    test_name = plot_utils.get_real_name(args.plot_type)
    # fig.suptitle('{0:s}, Bias Length = {1:d}'.format(test_name, args.bias_len))

    # Write single legend at top of figure
    handles, labels = axes[0,0].get_legend_handles_labels()
    for handle in handles:
        handle._sizes = [30]

    if args.plot_type == 'bias':
        pass

    if args.plot_type == 'budget':
        handles = handles[1:]
        labels = labels[1:]

    labels.append('Stained')

    pad = 5
    n_cols = len(labels)
    # n_cols = 2
    legend = axes[0,0].legend(handles, labels, frameon=True,
            bbox_to_anchor=(-0.1, 1.4), loc='upper left', ncol=n_cols)
    annotations.append(legend)

    # title_text = r'{}, $|F| = {}$'.format(test_name, args.bias_len)
    # title = plt.annotate(title_text, xy=(0.5, 1.0), xytext=(0, -50),
    #         xycoords='figure fraction', textcoords='offset points',
    #         size='large', ha='center', va='baseline')
    # annotations.append(title)

    # Add annotations along top specifying model
    for ax, model_name in zip(axes[0], models):
        col = plot_utils.get_real_name(model_name)
        a = ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')
        annotations.append(a)

    # Add annotations along side specifying dataset
    for ax, dataset_name in zip(axes[:,0], datasets):
        row = plot_utils.get_real_name(dataset_name)
        a = ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center', rotation=90)
        annotations.append(a)

    # Final adjustments and save plot
    if args.plot_type == 'bias':
        # fig.subplots_adjust(top=0.85, bottom=0.05, left=0.08, right=0.95,
        #         hspace=0.2, wspace=0.2)
        fig.subplots_adjust(top=0.80, bottom=0.05, left=0.08, right=0.95,
                hspace=0.2, wspace=0.2)

    elif args.plot_type == 'budget':
        fig.subplots_adjust(top=0.85, bottom=0.1, left=0.08, right=0.95,
                hspace=0.25, wspace=0.25)

    logger.info('Saving plot to: {}'.format(args.output))
    plt.savefig(
        args.output,
        bbox_extra_artists=annotations,
        bbox_inches='tight',
        shadow=True,
        format='pdf')


def bias_plot(df, datasets, models, bias_len):
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.5})
    fig, axes = plot_utils.get_subplots(datasets, models, sharex=False, sharey=False)

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            # Select rows for this (dataset, model) pair
            mask  = df['Dataset'] == dataset
            mask &= df['Model'] == model
            mask &= df['Bias Length'] == bias_len
            data = df[mask]

            # no_data_msg = "No rows found for {}, {}".format(dataset, model)
            # assert (not data.empty), no_data_msg

            if data.empty: continue

            ax = sns.barplot(data=data, x='Model Bias', y='Accuracy',
                hue='Region', hue_order=[r'$R_{orig}$', r'$\neg R$'], ax=axes[i, j])

            for b, bar in enumerate(ax.patches):
                if b % 2 == 1:
                    bar.set_hatch('/')
            # exit()

            # Remove individual subplot legends in favor of a single global one
            ax.get_legend().remove()
            ax.set_ylim(0, 1.0)
            ticks = ax.get_yticks()
            ax.get_yaxis().set_ticks(ticks[1:])

            ax.set_xlabel('Model', visible=True)

            if i != len(datasets) - 1:
                ax.set_xlabel('', visible=False)

            if j != 0:
                ax.set_ylabel('', visible=False)

    return fig, axes


def budget_plot(df, datasets, models, explainers, bias_len):
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.5})

    fig, axes = plot_utils.get_subplots(datasets, models, sharex=False,
            sharey=False)

    # Fill grid of datasets x models
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            mask  = df['Dataset'] == dataset
            mask &= df['Model'] == model
            mask &= df['Bias Length'] == bias_len
            mask &= df['Budget'].isin(BUDGETS)
            data = df[mask]

            # Get palette, but replace with hardcoded colors
            pal = sns.color_palette()
            for n, exp in enumerate(explainers):
                pal[n] = EXPLAINER_COLORS[exp]
            pal = np.array(pal)

            image_method = 'Intersect %'
            # image_method = 'Intersect % (circle r=5)'
            # image_method = 'Intersect % (circle r=10)'
            # image_method = 'Intersect % (circle r=15)'
            is_image = image_method in df

            if is_image:
                y_label = image_method
                data.loc[:, y_label] = data.loc[:, y_label].astype(float)
                data.loc[:, y_label] *= 100.0
            else:
                y_label = 'Recall'

            # Actual plot call
            markers = ["s", "^", "o", "P", "X", "D"]
            ax = sns.lineplot( data=data, x='Budget', y=y_label, hue='Explainer',
                    hue_order=explainers, style='Explainer', style_order=explainers,
                    err_style='bars', markers=markers, dashes=False, ax=axes[i, j],
                    palette=pal, ms=10, alpha=0.75)

            # Remove subplot legends in favor of a single one
            ax.get_legend().remove()
            if not is_image:
                ax.get_xaxis().set_ticks(sorted(data['Budget'].unique()))
                ax.set_ylim(0, 1.1)
            else:
                ax.set_ylim(0, 100)

            if i != len(datasets) - 1:
                ax.set_xlabel('', visible=False)

            if j != 0:
                ax.set_ylabel('', visible=False)

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
    parser.add_argument(
        '-b', '--bias-len',
        type=int,
        metavar='bias_len',
        required=False,
        default=BIAS_LENGTH,
        help='bias length to plot (default = {})'.format(BIAS_LENGTH))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
