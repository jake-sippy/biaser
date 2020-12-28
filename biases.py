import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Error if region R has less than this many examples
MIN_BIASED_EXAMPLES = 10

class TooManyAttemptsError(Exception):
    pass

class Bias:
    # Superclass

    def bias(self, instances, labels, runlog):
        # return (biased_labels, was_biased)
        raise NotImplemented


    def build_df_from_df(self, df, runlog):
        columns = ['reviews', 'label_orig', 'label_bias', 'biased', 'flipped']
        reviews = df['reviews'].values
        labels_orig = df['labels'].values
        labels_bias, biased, flipped = self.bias(reviews, labels_orig, runlog)
        data = zip(reviews, labels_orig, labels_bias, biased, flipped)
        return pd.DataFrame(data=data, columns=columns)


    def build_df(self, reviews, labels_orig, runlog):
        columns = ['reviews', 'label_orig', 'label_bias', 'biased', 'flipped']
        labels_bias, biased, flipped = self.bias(reviews, labels_orig, runlog)
        data = zip(reviews, labels_orig, labels_bias, biased, flipped)
        return pd.DataFrame(data=data, columns=columns)


class ComplexBias(Bias):
    def __init__(self, reviews, labels, bias_len, min_df, max_df, runlog,
            tokenizer=None, quiet=False):
        self.reviews = reviews
        self.labels = labels
        self.bias_len = bias_len
        self.min_df = min_df
        self.max_df = max_df
        self.runlog = runlog
        self.quiet = quiet

        # Build vocab that appears > min_df and < max_df
        if not self.quiet: print('Creating bias...')
        self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=True,
                preprocessor=None,
                tokenizer=tokenizer,
                stop_words=None,
                ngram_range=(1, 1),
                analyzer='word',
                min_df=min_df,
                max_df=max_df,
                binary=True,
        )
        self.vectorizer.fit(reviews)
        self.feature_names = self.vectorizer.get_feature_names()
        runlog['bias_attempts'] = 0
        self._load_bias(runlog)

    def _load_bias(self, runlog):
        idxs = np.arange(len(self.feature_names))
        self.bias_idxs = np.random.choice(idxs, self.bias_len, replace=False)
        self.bias_words = [self.feature_names[i] for i in self.bias_idxs]
        runlog['bias_words'] = self.bias_words
        runlog['bias_len'] = self.bias_len
        runlog['bias_attempts'] += 1
        if not self.quiet: print('\tBIAS_ATTEMPT = {}'.format(runlog['bias_attempts']))
        if not self.quiet: print('\tBIAS_WORDS = {}'.format(self.bias_words))
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.bias_label = unique_labels[np.argmin(counts)]

    def bias(self, instances, labels, runlog):
        vec = self.vectorizer.transform(instances)
        bias_labels = []
        biased = []
        flipped = []
        for i in range(vec.shape[0]):
            instance = vec[i].toarray()[0]
            if np.all(instance[self.bias_idxs] > 0):
                bias_labels.append(self.bias_label)
                biased.append(True)
                # True if label changed
                flipped.append(self.bias_label != labels[i])
            else:
                bias_labels.append(labels[i])
                biased.append(False)
                flipped.append(False)

        R_size = np.sum(biased)
        if R_size < MIN_BIASED_EXAMPLES:
            if runlog['bias_attempts'] < 3:
                self._load_bias(runlog)
                return self.bias(instances, labels, runlog)
            else:
                msg = 'Exceded 3 attempts to create bias - R too small'
                assert False, msg
                raise TooManyAttemptsError(msg)

        return bias_labels, biased, flipped


MIN_OCCUR = 0.20    # Min. attribute occurance to consider for selection
MAX_OCCUR = 0.50    # Max. attribute occurance to consider for selection
MIN_CERTAINTY = 3   # Min. user certainty that attribute is pictured (1-4)
FLIP_NR = False     # Flip examples without the attr. to the opposite class
BIAS_CLASS = 1      # Class to set biased example to

from image_utils import NUM_ATTRS

class BirdBias(Bias):
    # self.bias_attr is stored as 1-indexed to align with attributes.txt
    # and the names of the attribute columns within the BirdDataset

    def __init__(self, dataset, attr_id, runlog):
        # Drop N/A, i.e., attributes that dont have an accompanying part id
        attr_parts = dataset.attribute_parts.dropna()
        all_attrs = dataset.data[ [int(i) for i in range(1, NUM_ATTRS + 1)] ]
        all_attrs = all_attrs.to_numpy()
        valid_ids = attr_parts['attr_id'].unique()
        valid_attrs = np.zeros_like(all_attrs)
        valid_attrs[:, valid_ids] = all_attrs[:, valid_ids]

        totals = np.sum(valid_attrs >= MIN_CERTAINTY, axis=0)
        occurs = totals / valid_attrs.shape[0]


        if attr_id is not None:
            print('\t** BIAS MANUALLY SET **')
            attribute = attr_parts[ attr_parts['attr_id'] == attr_id ].iloc[0]
            occurance = occurs[attr_id - 1]
        else:
            print('\tMIN OCCUR = {: >.2f}'.format(MIN_OCCUR))
            print('\tMAX OCCUR = {: >.2f}'.format(MAX_OCCUR))
            print('\tCERTAINTY >= {: >3d}'.format(MIN_CERTAINTY))

            common_attrs = []
            for i, occurance in enumerate(occurs):
                if MIN_OCCUR <= occurance <= MAX_OCCUR:
                    common_attrs.append( (i + 1, occurance) )

            print('\t# RESULTS = {: >4d}'.format(len(common_attrs)))
            assert len(common_attrs) > 0, 'No attributes meet requirements'

            # i = np.random.choice(np.arange(len(common_attrs)))
            i = int( runlog['seed'] )
            attr_id, occurance = common_attrs[i]
            attribute = attr_parts[ attr_parts['attr_id'] == attr_id ].iloc[0]

        self.attr_id    = attr_id
        self.bias_label = BIAS_CLASS
        self.part_id    = attribute['part_id']
        self.attr_name  = attribute['attr_name']
        self.part_name  = attribute['part_name']

        runlog['bias_attr']      = int(self.attr_id)
        runlog['bias_attr_name'] = str(self.attr_name)
        runlog['bias_part']      = int(self.part_id)
        runlog['bias_part_name'] = str(self.part_name)
        runlog['bias_label']     = int(self.bias_label)

        print()
        print('\tBIAS_ATTR = {: >4d} ({})'.format(self.attr_id, self.attr_name))
        print('\tBIAS_PART = {: >4d} ({})'.format(self.part_id, self.part_name))
        print('\tOCCURANCE = {: >.2f}'.format(occurance))


    def bias(self, attributes, label):
        if attributes[self.attr_id - 1] >= MIN_CERTAINTY:
            biased = True
            flipped = self.bias_label != label
            return self.bias_label, biased, flipped

        return label, False, False
