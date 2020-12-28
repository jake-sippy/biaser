import os
import sys
import json
import time
import copy
import shutil
import argparse
from multiprocessing import Pool
from allennlp.training.trainer import Trainer
from allennlp.commands.train import train_model_from_file
from sklearn.model_selection import train_test_split

import tqdm
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import utils
import biases
import models
import plot_utils

from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from bertmodel import RobertaLarge, TRANSFORMER_WORDPIECE_LIMIT

from image_utils import BirdDataset, evaluate_models

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

import explainers


# GLOBALS ######################################################################
global args                     # Arguments from cmd line
LOG_PATH = 'logs'               # Top level directory for log files
DATA_DIR = 'datasets'           # Folder containing datasets
TRAIN_SIZE = 0.6                # Train split ratio
VALID_SIZE = 0.2                # Validation split ratio
TEST_SIZE  = 0.2                # Test split ratio
BIAS_MIN_DF = 0.20              # Min occurance for words to be bias words
BIAS_MAX_DF = 0.60              # Max occurance for words to be bias words
MAX_BUDGET = 5                  # Upper bound of budget to test explainers
N_SAMPLES = 50                  # Number of samples to evaluate each explainer
N_BAGS = 3                      # Number of bags to create in bagging test
BIAS_LENS = range(1, 2)         # Range of bias lengths to run


# TODO genrerlize this file, bert_run, and run.py into single main
# RESNET specific changes

MODEL_TYPE = 'resnet152'
# MODEL_TYPE = 'mnasnet'

# DATASET_NAME = 'cub200_warbler_sparrow'
DATASET_NAME = 'cub200_gull_wren'

INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50

NORMALIZE_MEANS = [0.485, 0.456, 0.406]
NORMALIZE_STDS  = [0.229, 0.224, 0.225]

SERIAL_DIR = 'image_classifiers'

# Train an unstained model as a control if True
TRAIN_ORIG = False

# Test types handled by this script
TESTS = [ 'bias_test', 'budget_test' ]

# Budget test constants
BUDGET_MIN = 5
BUDGET_MAX = 6
BUDGET_STEP = 1
NUM_EXPLAIN = 50

# Saving explanations over images
SAVE_BUDGET_IMAGES = True
MASK_OPACITY = 0.4
EXPLAINER_EXAMPLE_DIR = 'explainer_examples'


def main():
    os.environ['KMP_WARNINGS'] = '0'
    if args.device >=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    if args.quiet: sys.stdout = open(os.devnull, 'w')
    pool_size = args.n_workers
    seeds = range(args.seed_low, args.seed_high)

    datasets = []
    if args.toy:    # only test on 1 small dataset
        datasets.append(TOY_DATASET)
    else:
        for filename in os.listdir(args.data_dir):
            datasets.append(os.path.join(args.data_dir, filename))

    # Build list of arguments
    arguments = []
    for seed in range(args.seed_low, args.seed_high):
        # for dataset in datasets:
            for bias_len in BIAS_LENS:
                arguments.append({
                    'seed': seed,
                    'dataset': DATASET_NAME,
                    'model_type': MODEL_TYPE,
                    'bias_length': bias_len
                })

    if pool_size == 1:
        for arg in arguments:
            run_seed(arg)
    else:
        pool = Pool(pool_size, maxtasksperchild=1)
        imap_results = pool.imap(run_seed, arguments, chunksize=1)
        list(tqdm.tqdm(imap_results, total=len(arguments)))
        pool.close()
        pool.join()


def run_seed(arguments):
    print(arguments)
    seed = arguments['seed']
    model_type = arguments['model_type']
    bias_length = arguments['bias_length']
    dataset_name = arguments['dataset']

    if dataset_name == 'cub200_warbler_sparrow':
        classes = ['Warbler', 'Sparrow']
    elif dataset_name == 'cub200_gull_wren':
        classes = ['Gull', 'Wren']
    else:
        assert False, 'Unknown dataset name: ' + dataset_name

    # Set random state across libraries
    print()
    print( ('-' * 30) + ' SEED: ' + str(seed) + ' ' + ('-' * 30) )
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Building Runlog dictionary with seed args
    runlog = {}
    runlog['toy']        = args.toy
    runlog['seed']       = seed
    runlog['test_name']  = args.test
    runlog['model_type'] = model_type
    runlog['dataset']    = dataset_name
    runlog['bias_len']   = bias_length

    orig_save_path = os.path.join(SERIAL_DIR, dataset_name, model_type, 'orig_model',
                'model_' + str(seed) + '.torch')
    bias_save_path = os.path.join(SERIAL_DIR, dataset_name, model_type, 'bias_model',
                'model_' + str(seed) + '.torch')

    data_transforms = {
        'train': A.Compose([
            A.RandomResizedCrop(INPUT_SIZE, INPUT_SIZE),
            A.HorizontalFlip(),
            A.Normalize(NORMALIZE_MEANS, NORMALIZE_STDS),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy')),
        'val': A.Compose([
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.CenterCrop(INPUT_SIZE, INPUT_SIZE),
            A.Normalize(NORMALIZE_MEANS, NORMALIZE_STDS),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy')),
    }

    train_data = BirdDataset('train', data_transforms['train'], None, True, classes)

    if args.test == 'bias_test':
        bias_model = models.PretrainedModels(2, model_type)

        if os.path.exists(bias_save_path) and not args.force:
            # Load saved stained model
            print('\tSAVED MODEL FOUND AT: {}'.format(bias_save_path))
            attr_id = bias_model.load(bias_save_path)
            print('\nLoading stain...')
            biaser = biases.BirdBias(train_data, attr_id, runlog)

        else:
            # Train new stained model
            print('\nGenerating stain...')
            biaser = biases.BirdBias(train_data, None, runlog)
            dataloaders_dict = {
                x: torch.utils.data.DataLoader(
                    BirdDataset(x, data_transforms[x], biaser, True, classes),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
                for x in ['train', 'val']
            }

            print('\nTraining biased model...')
            bias_model.fit(dataloaders_dict, runlog, num_epochs=NUM_EPOCHS, bias=True)
            bias_model.save(bias_save_path, biaser.attr_id)
            print('\tMODEL SAVED TO: {}'.format(bias_save_path))

        # Train an unstained model for comparison
        orig_model = None
        if TRAIN_ORIG:
            print('\nTraining original model...')
            orig_model = models.PretrainedModels(2, model_type)
            if os.path.exists(orig_save_path) and not args.force:
                print('\tSAVED MODEL FOUND AT {}'.format(orig_save_path))
                orig_model.load(orig_save_path)
            else:
                orig_model.fit(dataloaders_dict, runlog, num_epochs=NUM_EPOCHS, bias=False)
                orig_model.save(orig_save_path, biaser.attr_id)
                print('\tMODEL SAVED TO: {}'.format(orig_save_path))

        # Evaluate the models on test data
        transform = data_transforms['val']
        test_dataset = BirdDataset('test', transform, biaser, True, classes)
        evaluate_models(orig_model, bias_model, test_dataset, runlog)

        if not args.no_log:
            utils.save_log(args.log_dir, runlog)
        return


    elif args.test == 'budget_test':
        print('\nLoading biased model...')
        bias_model = models.PretrainedModels(2, model_type)
        if os.path.exists(bias_save_path):
            print('\tSAVED MODEL FOUND AT {}'.format(bias_save_path))
            attr_id = bias_model.load(bias_save_path)
        else:
            print('\tNO MODEL FOUND AT {}'.format(bias_save_path))
            print('\tPLEASE RUN "bias_test" FIRST FOR THIS SEED')
            return

        print('\nLoading stain...')
        biaser = biases.BirdBias(train_data, attr_id, runlog)

        transform = data_transforms['val']
        train_dataset = BirdDataset('train', transform, biaser, True, classes)
        test_dataset = BirdDataset('test', transform, biaser, True, classes)
        evaluate_models(None, bias_model, test_dataset, runlog)

        if model_type == 'resnet152':
            target_layer = 'layer4'
            # exit()
        elif model_type == 'mnasnet':
            target_layer = 'layers.14'
        else:
            assert False

        # TODO test if this helps other grad methods, it's required for gradcam
        bias_model.grad_all()

        label = biaser.bias_label
        explainers_to_test = {
            'grad': explainers.VanillaGradExplainer(bias_model, label),
            'smooth': explainers.SmoothGradExplainer(bias_model, label),

            'LIME': explainers.LimeImageExplainer(bias_model, label),
            'random': explainers.RandomImageExplainer(),

            # 'SHAP': explainers.ShapImageExplanier(bias_model, label, train_dataset),
            'Grad-CAM': explainers.GradCamExplainer(bias_model, target_layer, label),
        }

        print('\nTesting explainers...')

        test_examples = []
        for i in range(len(test_dataset)):
            if test_dataset[i]['flipped']:
                pred = bias_model.predict(test_dataset[i]['image'])
                if pred.item() == test_dataset[i]['bias_label']:
                    test_examples.append(test_dataset[i])

                # test_examples.append(test_dataset[i])

                if len(test_examples) >= NUM_EXPLAIN:
                    break

        num_explain = len(test_examples)
        for name, explainer in explainers_to_test.items():
            runlog['explainer'] = name
            print('\n\tEXPLAINER = {}'.format(name))

            start = BUDGET_MIN
            end = BUDGET_MAX + 1
            step = BUDGET_STEP
            if name == 'LIME' and BUDGET_STEP < 5:
                start = 5
                step = 5

            for budget in range(start, end, step):
                recall = 0.0
                intersect_circle = 0.0
                intersect_segment = 0.0

                print('\tEST. BUDGET = {}'.format(budget))
                for i in tqdm.tqdm(range(num_explain), total=num_explain):
                    img_id = int(test_examples[i]['img_id'])
                    image = test_examples[i]['image']
                    path = test_examples[i]['path']
                    x = test_examples[i]['part_x']
                    y = test_examples[i]['part_y']

                    runlog['example_id'] = i
                    runlog['part_x'] = x
                    runlog['part_y'] = y
                    runlog['img_id'] = img_id
                    runlog['img_path'] = path
                    runlog['orig_label'] = int(test_examples[i]['label'])
                    runlog['bias_label'] = int(test_examples[i]['bias_label'])

                    explain_mask = explainer.explain(image, budget)
                    true_budget = np.sum(explain_mask) / np.prod(explain_mask.shape)
                    runlog['budget'] = int(round(true_budget * 100))

                    # Scoring methods

                    # Recall
                    recalled = explain_mask[y, x] > 0
                    recall += 1.0 if recalled else 0.0
                    runlog['recalled'] = bool(recalled)
                    runlog['recall'] = 1.0 if recalled else 0.0

                    def score(explain_mask, gt_mask):
                        intersect_mask = np.zeros_like(gt_mask)
                        intersect = (explain_mask == 1.0) & (gt_mask == 1.0)
                        intersect_mask[intersect] = 1.0
                        total_correct = int(np.sum(intersect_mask))
                        total_area = int(np.sum(gt_mask))
                        return total_correct / total_area

                    # Ground Truth #1 (Circle near click location)
                    radii = {'small': 5, 'medium': 10, 'large': 15}
                    for size in radii:
                        circle_mask = np.zeros_like(explain_mask)
                        for yi in range(circle_mask.shape[0]):
                            for xi in range(circle_mask.shape[1]):
                                in_circle = (x - xi)**2 + (y - yi)**2 < radii[size]**2
                                circle_mask[yi, xi] = 1.0 if in_circle else 0.0

                        res = score(explain_mask, circle_mask)
                        runlog['radius_' + size] = radii[size]
                        runlog['intersect_percentage_circle_' + size] = res
                    intersect_circle += res

                    # Ground Truth #2 (Segment near click location)
                    segments = slic(
                            np.moveaxis(np.array(image), 0, -1),
                            n_segments=100,
                            start_label=0,
                            sigma=1.0,
                            slic_zero=True
                    )
                    superpixel_idx = segments[y, x]
                    segment_mask = np.zeros_like(segments, dtype=np.float32)
                    segment_mask[ segments == superpixel_idx ] = 1.0

                    res = score(explain_mask, segment_mask)
                    intersect_segment += res
                    runlog['intersect_percentage_segment'] = res

                    if not args.no_log:
                        utils.save_log(args.log_dir, runlog, quiet=True)

                    # Save this example as an image with highlighted areas
                    if SAVE_BUDGET_IMAGES:
                        gt_mask = segment_mask
                        gt_method = 'segment'
                        save_image_with_masks(image, explain_mask, gt_mask,
                                gt_method, runlog)

                recall /= num_explain
                intersect_circle /= num_explain
                intersect_segment /= num_explain

                print('\tAVG RECALL                    = {:.4f}'
                        .format(recall))
                print('\tAVG INTERSECTION (circle Lg.) = {:.4f}'
                        .format(intersect_circle))
                print('\tAVG INTERSECTION (segment)    = {:.4f}'
                        .format(intersect_segment))

        return


def save_image_with_masks(image, explain_mask, gt_mask, gt_method, runlog):
    assert gt_method in ['circle', 'segment']

    example_id  = runlog['example_id']
    explainer   = runlog['explainer']
    model       = runlog['model_type']
    dataset     = runlog['dataset']
    budget      = runlog['budget']
    seed        = runlog['seed']
    recalled    = runlog['recalled']
    intersect   = runlog['intersect_percentage_' + gt_method]
    attr_name   = runlog['bias_attr_name']
    x = runlog['part_x']
    y = runlog['part_y']

    out_dir = os.path.join(
            EXPLAINER_EXAMPLE_DIR,
            dataset,
            model,
            explainer,
            'seed_{:02d}_{}'.format(seed, attr_name),
            'budget_{:03d}'.format(budget),
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_filename = os.path.join(out_dir,
            'test_picture_{:02d}.png'.format(example_id))

    def mask_to_img(mask, color):
        rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
        r, g, b, a = cv2.split(rgba)
        r = np.zeros_like(r) if color not in ['red', 'yellow'] else r
        g = np.zeros_like(r) if color not in ['green', 'yellow' 'cyan'] else g
        b = np.zeros_like(r) if color not in ['blue', 'cyan'] else b
        a = mask * MASK_OPACITY
        return cv2.merge( (r,g,b,a) )

    image = image.double().detach().cpu().numpy()
    image = np.moveaxis(image, 0, -1)
    image = (image * NORMALIZE_STDS) + NORMALIZE_MEANS
    image = np.clip(image, 0.0, 1.0)

    intersect_mask = np.zeros_like(gt_mask)
    intersection = (explain_mask == 1.0) & (gt_mask == 1.0)
    intersect_mask[ intersection ] = 1.0

    # Hide in favor of intersection mask
    explain_mask[ intersection ] = 0.0
    gt_mask[ intersection ] = 0.0

    explain_img = mask_to_img(explain_mask, 'red')
    gt_img = mask_to_img(gt_mask, 'blue')
    intersect_img = mask_to_img(intersect_mask, 'green')

    # legend_entries = [
    #     Patch(facecolor='yellow', label='Explanation'),
    #     Patch(facecolor='blue', label='Ground Truth'),
    # ]

    sns.set_context("paper", font_scale=2.0) #, rc={"lines.linewidth": 2.5})
    plt.tick_params(
        axis='both',
        which='both',
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelbottom=False,
        labelleft=False)
    plt.imshow(image)
    plt.imshow(gt_img)
    plt.imshow(explain_img)
    plt.imshow(intersect_img)
    # leg = plt.legend(handles=legend_entries, loc='upper right')

    plt.title('{:s}, {:.1%}'.format(
            plot_utils.get_real_name(explainer), intersect))

    plt.savefig(out_filename, bbox_inches='tight')
            # bbox_to_anchor=(1.05, 1), bbox_extra_artists=[leg])
    plt.clf()


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'test', type=str, metavar='TEST',
            help=' | '.join(TESTS))
    parser.add_argument( 'seed_low', type=int, metavar='SEED_LOW',
            help='Lower bound of seeds to loop over (inclusive)')
    parser.add_argument( 'seed_high', type=int, metavar='SEED_HIGH',
            help='Higher bound of seeds to loop over (exclusive)')
    parser.add_argument( 'n_workers', type=int, metavar='N_WORKERS',
            help='Number of workers to spawn')
    parser.add_argument( '--log-dir', type=str, metavar='LOG_DIR', default=LOG_PATH,
            help='Log file directory (default = {})'.format(LOG_PATH))
    parser.add_argument( '--data-dir', type=str, metavar='DATA_DIR', default=DATA_DIR,
            help='Dataset directory (default = {})'.format(DATA_DIR))
    parser.add_argument( '--quiet', action='store_true',
            help='Do not print out information while running')
    parser.add_argument( '--no-log', action='store_true',
            help='Do not log information while running')
    parser.add_argument( '--single-thread', action='store_true',
            help='Force single-thread for multiple seeds')
    parser.add_argument( '--toy', action='store_true',
            help='Run a toy version of the test')
    parser.add_argument( '--force', action='store_true',
            help='Force a new trained model')
    parser.add_argument( '--device', type=int, default=-1, metavar='DEVICE',
            help='CUDA device to use (default = -1)')
    parser.add_argument( '--serial-dir', type=str, default=SERIAL_DIR, metavar='SERIAL',
            help='Directory to serialize trained models to')

    args = parser.parse_args()

    bad_test_msg = 'Test not found: {}'.format(args.test)
    assert (args.test in TESTS), bad_test_msg
    bad_seed_msg = 'No seeds in [{}, {})'.format(args.seed_low, args.seed_high)
    assert (args.seed_low < args.seed_high), bad_seed_msg

    return args


if __name__ == '__main__':
    args = setup_args()
    main()
