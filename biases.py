import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Error if region R has less than this many examples
MIN_BIASED_EXAMPLES = 10

class Bias:
    def bias(self, instances, labels):
        # return (biased_labels, was_biased)
        raise NotImplemented

    def build_df(self, reviews, labels_orig):
        columns = ['reviews', 'label_orig', 'label_bias', 'biased', 'flipped']
        labels_bias, biased, flipped = self.bias(reviews, labels_orig)
        data = zip(reviews, labels_orig, labels_bias, biased, flipped)
        return pd.DataFrame(data=data, columns=columns)


class ComplexBias(Bias):
    def __init__(self, reviews, labels, bias_len, min_df, max_df, runlog,
            quiet=False):
        if not quiet: print('Creating bias...')
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
        idxs = np.arange(len(self.feature_names))
        self.bias_idxs = np.random.choice(idxs, bias_len, replace=False)
        self.bias_words = [self.feature_names[i] for i in self.bias_idxs]

        runlog['bias_words'] = str(self.bias_words)
        runlog['bias_len'] = bias_len
        if not quiet: print('\tBIAS_WORDS = {}'.format(self.bias_words))
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.bias_label = unique_labels[np.argmin(counts)]

    def bias(self, instances, labels):
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
        assert np.sum(biased) > MIN_BIASED_EXAMPLES,\
                'Too few biased examples, decrease bias length'
        return bias_labels, biased, flipped



