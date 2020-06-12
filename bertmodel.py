import json
import os
from overrides import overrides
from typing import Dict
import argparse
import random

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.interpret.saliency_interpreters import IntegratedGradient, SimpleGradient

from allennlp.predictors.predictor import Predictor

from lime.lime_text import LimeTextExplainer

import tqdm

TRANSFORMER_WORDPIECE_LIMIT = 512
LSTM_MODEL_PATH = "https://s3-us-west-2.amazonaws.com/allennlp/models/sst-2-basic-classifier-glove-2019.06.27.tar.gz"
ROBERTA_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.02.17.tar.gz"


class RobertaLarge:

    def __init__(self,
            model_path=None,
            cuda_device=-1):
        # model_path = model_path or LSTM_MODEL_PATH
        model_path = model_path or ROBERTA_MODEL_PATH
        self.predictor = Predictor.from_path(model_path,
                cuda_device=cuda_device)

        _tokenizer = PretrainedTransformerTokenizer(model_name="roberta-base",
                max_length=TRANSFORMER_WORDPIECE_LIMIT)
        class_name_mapper = {"0": "Negative", "1": "Positive"}
        _model = self.predictor._model
        _label_namespace = _model._label_namespace
        class_names = [
            class_name_mapper[_model.vocab.get_index_to_token_vocabulary(_label_namespace).get(0)],
            class_name_mapper[_model.vocab.get_index_to_token_vocabulary(_label_namespace).get(1)]
        ]
        # reset the tokenizer to remove separators
        self.tokenizer = lambda s: [t.text.replace("Ġ", "").replace('ĉ', "") for t in _tokenizer.tokenize(s)][1:-1]
        self.explainer_lime = LimeTextExplainer(class_names=class_names) # , split_expression=self.tokenizer
        self.explainer_integrate = IntegratedGradient(self.predictor)
        self.explainer_simple = SimpleGradient(self.predictor)

    def predict(self, docs):
        """
        docs: list of strings
        """
        pred_list = []
        for doc in tqdm.tqdm(docs):
            instance = self.predictor._dataset_reader.text_to_instance(doc)
            pred_list.append(int(self.predictor.predict_instance(instance)["label"]))
        return pred_list

    def predict_proba(self, docs):
        prob_list = []
        for doc in tqdm.tqdm(docs):
            instance = self.predictor._dataset_reader.text_to_instance(doc)
            prob_list.append(self.predictor.predict_instance(instance)["probs"])
        return np.vstack(prob_list)

    def explain(self, sentence, use_simple=False):
        # sentence must be of type str - a single str input
        # There is probably a cleaner way to grab the tokens, need to check why saliency_interpret... isn't returning tokens?
        #tokens = [t.text for t in self.predictor.json_to_labeled_instances({"sentence": sentence})[0].fields["tokens"].tokens]
        #salience = self.explainer.saliency_interpret_from_json({"sentence": sentence})['instance_1']['grad_input_1']
        tokens = self.tokenizer(sentence)
        if use_simple:
            explainer = self.explainer_simple
        else:
            explainer = self.explainer_integrate
        salience = explainer.saliency_interpret_from_json({"sentence": sentence})['instance_1']['grad_input_1'][1:-1]
        return self._segment_with_tokens(
            sentence,
            [(tokens[i], salience[i]) for i in range(len(tokens))]
        )

    def _explain_lime_raw(self, sentence, num_features=None, num_samples=5000):
        if num_features == None:
            num_features = len(self.tokenizer(sentence))
        return self.explainer_lime.explain_instance(
            sentence, self.predict_proba, top_labels=2, num_features=num_features, num_samples=num_samples) #

    def explain_lime(self, sentence, num_samples=5000):
        # get the prediction
        exp = self._explain_lime_raw(sentence, num_samples=num_samples)
        ## NOT USING self.predict_proba because I need label idx that matches the max probability, not label.
        predicted = np.argmax(self.predict_proba([sentence])[0])
        # get the explanation for the selected class
        # get the tokens
        indexes = exp.as_map()[predicted]
        tokens = exp.as_list(label=predicted)
        tokens = [[indexes[i], t[0], t[1]] for i, t in enumerate(tokens)]
        tokens = sorted(tokens, key=lambda t: t[0])
        return self._segment_with_tokens(
            sentence,
            [t[1:] for t in tokens]
        )

    def _segment_with_tokens(self, text, token_weights):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        text_ptr = 0
        for token_weight in token_weights:
            token, weight = token_weight
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError("Tokenization produced tokens that do not belong in string!")
            text_ptr += len(token)
            if inter_token_string:
                list_form.append([len(list_form), ''.join(inter_token_string), 0])
            list_form.append([len(list_form), token, weight])
        if text_ptr < len(text):
            list_form.append([len(list_form), text[text_ptr:], 0])
        return list_form


def evaluate(model, dataset):
    #filename = f"../../flaskr/static/assets/{dataset}/data-task.json"
    filename = f"../../datasets/raw_data/human_ai/{dataset}/classifier/test2.json"
    test_data = json.load(open(filename))
    X = [r["X"] for r in test_data[:100]]
    Y = [r["Y"] for r in test_data[:100]]

    Y_pred = model.predict(X)
    #output_data = []
    #for i in range(len(test_data)):
    #    output_data.append({
    #            "X": X[i],
    #            "Y": Y[i],
    #            "pred": Y_pred[i],
    #            "temp-explain": model.explain(X[i])
    #            })
    # print(f"Dataset: {dataset}\tModel: AI2-pretrained")
    print(classification_report(Y, Y_pred))
    print("\n\n")
    #json.dump(output_data, open(os.path.join("temp_full.json"), "w"), indent=2)


@DatasetReader.register("custom_text")
class CustomTextDatasetReader(DatasetReader):
    def __init__(self,
            token_indexers: Dict[str, TokenIndexer]=None,
            balance_classes=False,
            **kwargs):
        super().__init__(**kwargs)
        # max_length ensures that we truncate the input
        self._tokenizer = PretrainedTransformerTokenizer(model_name="roberta-base",
                max_length=TRANSFORMER_WORDPIECE_LIMIT)
        self._token_indexers = token_indexers
        self.balance_classes = balance_classes

    @overrides
    def text_to_instance(self, doc, label=None):
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(doc)
        if len(tokens) == 0 or tokens is None:
            print("Data contains empty examples, needs fixing...")
            raise Exception
        fields["tokens"] = TextField(tokens,
                token_indexers=self._token_indexers)
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
            for i, r in enumerate(data):
                doc = r["X"]
                label = str(r["Y"])
                instance = self.text_to_instance(doc, label)
                if instance is not None:
                    yield instance


@DatasetReader.register("custom_text_csv")
class CustomTextDatasetReader(DatasetReader):
    def __init__(self,
            token_indexers: Dict[str, TokenIndexer]=None,
            balance_classes=False,
            **kwargs):
        super().__init__(**kwargs)
        # max_length ensures that we truncate the input
        self._tokenizer = PretrainedTransformerTokenizer(model_name="roberta-base",
                max_length=TRANSFORMER_WORDPIECE_LIMIT)
        self._token_indexers = token_indexers
        self.balance_classes = balance_classes

    @overrides
    def text_to_instance(self, doc, label=None):
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(doc)
        if len(tokens) == 0 or tokens is None:
            print("Data contains empty examples, needs fixing...")
            raise Exception
        fields["tokens"] = TextField(tokens,
                token_indexers=self._token_indexers)
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self, filepath):
        with open(filepath) as f:
            data = pd.read_csv(f, header=None, names=['reviews', 'labels'])
            for i, (idx, row) in enumerate(data.iterrows()):
                doc = row['reviews']
                label = str(row['labels'])
                instance = self.text_to_instance(doc, label)
                if instance is not None:
                    yield instance



def process_data(dirname):
    trainfile = os.path.join(dirname, "train.json")
    testfile = os.path.join(dirname, "test.json")

    def filter_examples(data):
        # remove X that are empty because the above dataset reader does not
        # check for this
        new_data = []
        for r in tqdm.tqdm(data):
            if len(r["X"]) > 2:
                new_data.append(r)
        return new_data

    def balance_examples(data):
        # NOTE(Gagan): Hard-coded for binary classification
        new_data = []
        class_count = [0, 0]
        for r in data:
            class_count[r["Y"]] += 1
        print(class_count)
        majority_class = int(class_count[1] > class_count[0])
        ratio = int(class_count[majority_class]/class_count[1-majority_class])
        print("Ratio:", ratio)
        for r in data:
            if r["Y"] != majority_class:  # duplicate examples of minority class
                new_data.extend([r for _ in range(ratio)])
            else:
                new_data.append(r)
        random.shuffle(new_data)
        return new_data


    with open(trainfile) as infile:
        data = balance_examples(filter_examples(json.load(infile)))
        n = len(data)
        split_idx = int(0.8 * n)
        train_data = data[:split_idx]
        print("New train size", len(train_data))
        val_data = data[split_idx:]
        print("New val size", len(val_data))
        json.dump(train_data, open(os.path.join(dirname, "train2.json"), "w"))
        json.dump(val_data, open(os.path.join(dirname, "val2.json"), "w"))

    with open(testfile) as infile:
        data = filter_examples(json.load(infile))
        print("New test size", len(data))
        json.dump(data, open(os.path.join(dirname, "test2.json"), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utilities for training/evaluating bert-based models.')
    parser.add_argument('command',
                    default='evaluate',
                    choices=['preprocess', 'evaluate'],
                    help="Choose between data preprocessing or demo of the latest version on clam")
    parser.add_argument('dataset',
                    default='beer',
                    choices=['beer', 'amzbook'],
                    help="Choose a dataset")
    parser.add_argument('--cuda_device',
                    default='-1',
                    type=int)
    args = parser.parse_args()

    if args.command == "preprocess":
        process_data(f"../../datasets/raw_data/human_ai/{args.dataset}/classifier/")
    elif args.command == "evaluate":
        model_path = f"https://clam.cs.washington.edu/global_static/{args.dataset}.tar.gz"
        model = RobertaLarge(model_path=model_path, cuda_device=args.cuda_device)
        evaluate(model, args.dataset)
