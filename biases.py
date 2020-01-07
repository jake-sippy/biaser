import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

MIN_BIASED_EXAMPLES = 10

class Bias:
    def bias(self, instances, labels):
        # Returns a tuple of biased labels and a boolean vector of which
        # instances were biased
        #
        # return (biased_labels, was_biased)
        raise NotImplemented


class NgramBias(Bias):
    def __init__(self, reviews, labels, ngrams, min_df, max_df, runlog):
        print('Creating random n-gram bias...')
        self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                stop_words=None,
                ngram_range=(ngrams, ngrams),
                analyzer='word',
                max_df=max_df,
                min_df=min_df,
                max_features=None,
                binary=True,
        )

        self.vectorizer.fit(reviews)
        self.feature_names = self.vectorizer.get_feature_names()
        assert len(self.feature_names) > 10, \
                'Not enough n-grams for n=%d' % ngrams
        self.bias_idx = np.random.randint(len(self.feature_names))
        self.ngram = self.feature_names[self.bias_idx]
        runlog['ngram'] = str(self.ngram)
        runlog['n'] = ngrams
        print('\tNGRAM = {}'.format(self.ngram))
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.bias_label = unique_labels[np.argmin(counts)]

    def bias(self, instances, labels):
        vec = self.vectorizer.transform(instances)
        bias_labels = []
        biased = []
        for i in range(vec.shape[0]):
            if vec[i, self.bias_idx] > 0:
                bias_labels.append(self.bias_label)
                biased.append(True)
            else:
                bias_labels.append(labels[i])
                biased.append(False)
        return bias_labels, biased


class ComplexBias(Bias):
    def __init__(self, reviews, labels, bias_len, min_df, max_df, runlog):
        print('Creating bias...')
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
                max_df=max_df,
                min_df=min_df,
                max_features=None,
                binary=True,
        )

        self.vectorizer.fit(reviews)
        self.feature_names = self.vectorizer.get_feature_names()
        words = len(self.feature_names)
        self.bias_idxs = np.random.choice(np.arange(words), bias_len,
                replace=False)
        self.bias_words = [self.feature_names[i] for i in self.bias_idxs]

        runlog['bias_words'] = str(self.bias_words)
        runlog['bias_len'] = bias_len
        print('\tBIAS_WORDS = {}'.format(self.bias_words))
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.bias_label = unique_labels[np.argmin(counts)]

    def bias(self, instances, labels):
        vec = self.vectorizer.transform(instances)
        bias_labels = []
        biased = []
        for i in range(vec.shape[0]):
            instance = vec[i].toarray()[0]
            if np.all(instance[self.bias_idxs] > 0):
                bias_labels.append(self.bias_label)
                biased.append(True)
            else:
                bias_labels.append(labels[i])
                biased.append(False)
        assert np.sum(biased) > MIN_BIASED_EXAMPLES,\
                'Too few biased examples, decrease bias length'
        return bias_labels, biased



