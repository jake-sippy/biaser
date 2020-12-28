import os
import time
import copy

import tqdm
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import utils

NUM_ATTRS = 312

class BirdDataset(Dataset):
    def __init__(self, mode, transform=None, bias=None, binary=True,
            binary_classes=['Warbler', 'Sparrow']):

        assert mode in ['train', 'val', 'test'], \
                'Unknown split passed to BirdDataset (%s)' % mode

        self.data_dir = 'CUB_200_2011/'
        self.mode = mode
        self.transform = transform
        self.binary = binary
        self.bias = bias

        # Build self.data DataFrame ############################################

        # Load img_id -> attr data
        attr_data = pd.read_csv(
            self.data_dir + 'attributes/image_attribute_labels.txt',
            sep=' ', header=None,
            names=['img_id', 'attr_id', 'is_present', 'certainty', 'time'],
            usecols=['img_id', 'attr_id', 'is_present', 'certainty'])

        attr_data = attr_data.pivot(index='img_id', columns='attr_id')
        attr_data['is_present'] = (2 * attr_data['is_present']) - 1
        attributes = attr_data['is_present'] * attr_data['certainty']

        # Load attributes -> parts data
        attribute_parts = pd.read_csv(
            self.data_dir + 'attributes_parts_test.txt', sep=' ', header=None,
            # self.data_dir + 'attributes_parts.txt', sep=' ', header=None,
            names=['attr_id', 'attr_name', 'part_name'])

        # Load part_id -> part_name
        part_ids = pd.read_csv(
                self.data_dir + 'parts/parts.txt', sep=' ', header=None,
                names=['part_id', 'part_name'])

        # Attr and parts (ids and names)
        self.attribute_parts = attribute_parts.merge(part_ids, on='part_name')

        # img_id -> img_path
        image_path_data = pd.read_csv(self.data_dir + 'images.txt',
            sep=' ', header=None, names=['img_id', 'img_path'])

        # Drop malformed examples
        #   - 5007 (user clicks are outside img dimensions)
        image_path_data = image_path_data[ image_path_data['img_id'] != 5007 ]

        data = image_path_data.merge(attributes, on='img_id')

        # Binarize data ########################################################
        if binary:
            assert len(binary_classes) == 2
            bird_labels = { binary_classes[0] : 0, binary_classes[1] : 1 }
            get_name = lambda row: row['img_path'].split('_')[-3]

            data['bird_name'] = data.apply(get_name, axis=1)
            data['label'] = data['bird_name'].map(bird_labels)
            data.dropna(inplace=True)
            data.reset_index(drop=True, inplace=True)
        else:
            # We will change this later using the bias object
            data['label'] = 0

        # Train / Val / Test Split #############################################
        train_test_data = pd.read_csv(self.data_dir + 'train_test_split_custom.txt',
            sep=' ', header=None, names=['img_id', 'is_train'])

        # 80/10/10 split
        is_train = data.merge(train_test_data, on='img_id')['is_train'] == 1

        if mode == 'train':
            data = data[is_train]
            data, _ = train_test_split(data, train_size=0.8888, random_state=1)
        elif mode == 'val':
            data = data[is_train]
            _, data = train_test_split(data, train_size=0.8888, random_state=1)
        elif mode == 'test':
            data = data[~is_train]
        data.reset_index(drop=True, inplace=True)

        # Handle staining function #############################################
        if bias is None:
            # No bias label to create, just clean up
            data.reset_index(drop=True, inplace=True)
            self.data = data
            return

        def apply_bias(row):
            attrs = row.loc[ list(range(1, NUM_ATTRS + 1)) ].to_numpy(dtype='int')
            label = row.loc['label']
            bias_label, biased, flipped = bias.bias(attrs, label)
            index = ['bias_label', 'biased', 'flipped']
            return pd.Series([bias_label, biased, flipped], index=index)

        bias_columns = data.apply(apply_bias, axis=1, result_type='expand')
        data['bias_label'] = bias_columns['bias_label']
        data['biased'] = bias_columns['biased']
        data['flipped'] = bias_columns['flipped']

        # Filter out biased examples with no part location
        part_data = pd.read_csv(
            self.data_dir + 'parts/part_locs.txt', sep=' ', header=None,
            names=['img_id', 'part_id', 'part_x', 'part_y', 'visible'],
            dtype='int')

        part_clicks = pd.read_csv(
            self.data_dir + 'parts/part_click_locs.txt', sep=' ', header=None,
            names=['img_id', 'part_id', 'part_x', 'part_y', 'visible', 'time'],
            usecols=['img_id', 'part_id', 'part_x', 'part_y', 'visible'],
            dtype='int')

        self.part_clicks = part_clicks[ part_clicks['part_id'] == bias.part_id ]

        part_data = part_data[ part_data['part_id'] == bias.part_id ]
        data = data.merge(part_data, on='img_id')
        error = data['biased'] & (data['visible'] == 0)
        data = data[ ~error ]
        data = utils.oversample(data, r_factor=2.0) if mode == 'train' else data
        data.reset_index(drop=True, inplace=True)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        img_id = row['img_id']
        attrs = row.loc[ list(range(1, NUM_ATTRS + 1)) ].to_numpy(dtype='int')
        label = row['label'].astype(int)
        path  = row['img_path']

        with open(self.data_dir + 'images/' + path, 'rb') as f:
            image_file = Image.open(f)
            image_rgb  = image_file.convert('RGB')
            image_rgb  = np.array(image_rgb)

        if self.bias is not None:
            visible = row['visible']
            keypoints = [ (row['part_x'], row['part_y']) ]
            # print(img_id)
            # print(keypoints[0])
            # print(image_rgb.shape)
            # print('-' * 10)
            transformed = self.transform(image=image_rgb, keypoints=keypoints)
            keypoints = np.array(transformed['keypoints'], dtype=np.int32)

            # During training, random cropping may remove keypoints, if there
            # is only 1, pass back 0,0
            if self.mode == 'train' and len(keypoints) == 0:
                keypoints = np.array( [(0, 0)] )

            return {
                'img_id': img_id,
                'image': transformed['image'],
                'attrs': attrs,
                'label': label,
                'path' : path,

                # bias specific
                'biased'    : row['biased'],
                'flipped'   : row['flipped'],
                'bias_label': row['bias_label'].astype(int),
                'part_x'    : int(round(keypoints[0][0])),
                'part_y'    : int(round(keypoints[0][1])),
                # 'clicks'    : click_keypoints,

                # Not really needed
                'attr_id'   : self.bias.attr_id,
                'attr_name' : self.bias.attr_name,
                'part_id'   : self.bias.part_id,
                'part_name' : self.bias.part_name,
            }

        else:
            transformed = self.transform(image=image_rgb, keypoints=[])
            return {
                'img_id': img_id,
                'image': transformed['image'],
                'attrs': attrs,
                'label': label,
                'path': path
            }



def evaluate_models(orig_model, bias_model, test_data, runlog):

    if orig_model is not None:
        orig_model.model.eval()
        print('\nEvaluating original and biased models...')
        models = {'orig': orig_model, 'bias': bias_model}
    else:
        print('\nEvaluating biased model...')
        models = {'bias': bias_model}

    bias_model.model.eval()

    for name, model in models.items():
        y_true = []
        y_pred = []
        biased = []
        length = len(test_data)
        for i in tqdm.tqdm(range(length), total=length):
            example = test_data[i]
            true = example['label' if name == 'orig' else 'bias_label']
            with torch.no_grad():
                pred = model.predict(example['image']).item()

            y_true.append(true)
            y_pred.append(pred)
            biased.append(example['biased'])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        biased = np.array(biased)

        acc    = float(accuracy_score(y_true, y_pred))
        f1     = float(f1_score(y_true, y_pred))
        acc_r  = float(accuracy_score(y_true[biased], y_pred[biased]))
        acc_nr = float(accuracy_score(y_true[~biased], y_pred[~biased]))

        runlog[name + '_test_acc'] = acc
        runlog[name + '_test_f1']  = f1
        runlog[name + '_R_acc']    = acc_r
        runlog[name + '_NR_acc']   = acc_nr

        print('\t{} MODEL TEST ACC:    {:.4f}'.format(name.upper(), acc))
        print('\t{} MODEL TEST F-1:    {:.4f}'.format(name.upper(), f1))
        print('\t{} MODEL R Accuracy:  {:.4f}'.format(name.upper(), acc_r))
        print('\t{} MODEL NR Accuracy: {:.4f}'.format(name.upper(), acc_nr))
        print()

    if orig_model is not None:
        # legacy (for plotting)
        runlog['results'] = [
            [runlog['orig_R_acc'], runlog['orig_NR_acc']],
            [runlog['bias_R_acc'], runlog['bias_NR_acc']],
        ]

