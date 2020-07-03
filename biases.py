import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Error if region R has less than this many examples
MIN_BIASED_EXAMPLES = 10

class TooManyAttemptsError(Exception):
    pass

class Bias:
    # Superclass

    def bias(self, instances, labels):
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
            quiet=False):
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
                tokenizer=None,
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



