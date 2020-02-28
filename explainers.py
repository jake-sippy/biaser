import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer as LimeText
from shap import KernelExplainer, DeepExplainer, kmeans


class Explainer:
    def __init__(self, model, training_data, seed, text=False):
        # Actual model must be last step
        self.training_data = training_data
        if text:
            self.model = model
        else:
            self.preprocessor = Pipeline(steps=model.steps[:-1])
            self.model = model.steps[-1][1]
        if 'counts' in model.named_steps:
            self.feature_names = model.named_steps['counts'].get_feature_names()
        elif 'text2ind' in model.named_steps:
            self.feature_names = model.named_steps['text2ind'].get_feature_names()
        else:
            assert False, 'Current model has no way to get feature names'
        self.seed = seed

    def explain(self, instance, budget):
        # Return a list of tuples: (feature, importance). Sorted in decreasing
        # order of importance
        raise NotImplementedError


class LimeExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(LimeExplainer, self).__init__(model, training_data, seed)
        data = self.preprocessor.transform(self.training_data)
        self.explainer = LimeTabularExplainer(
                training_data=data.astype(float),
                feature_names=self.feature_names,
                random_state=self.seed)

    def explain(self, instance, budget):
        instance = self.preprocessor.transform([instance])[0]
        exp = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=self.model.predict_proba,
                num_features=budget)

        feature_pairs = exp.as_map()[1]
        feats = []
        for feat_idx, importance in feature_pairs:
            feats.append( (self.feature_names[feat_idx], importance) )
        return feats[:budget]


class ShapExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(ShapExplainer, self).__init__(model, training_data, seed)
        data = self.preprocessor.transform(training_data)
        background_data = kmeans(data, 10)
        self.explainer = KernelExplainer(
                model=self.model.predict_proba,
                data=background_data)

    def explain(self, instance, budget):
        instance = self.preprocessor.transform([instance])[0]
        values = self.explainer.shap_values(
                X=instance,
                nsamples="auto",
                l1_reg='num_features(%d)' % budget)[1]
        pairs = sorted(
                zip(self.feature_names, values),
                key=lambda x : abs(x[1]),
                reverse=True
        )
        return pairs[:budget]


class GreedyExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(GreedyExplainer, self).__init__(model, training_data, seed)

    def explain(self, instance, budget):
        instance = self.preprocessor.transform([instance])
        inst = instance.reshape(1, -1)
        base = self.model.predict_proba(inst)[0, 1]
        values = []
        for i, val in enumerate(inst[0]):
            inst[0, i] = 0
            delta = self.model.predict_proba(inst)[0, 1]
            values.append(base - delta)
            inst[0, i] = val

        pairs = sorted(
                zip(self.feature_names, values),
                key=lambda x: abs(x[1]),
                reverse=True
        )
        return pairs[:budget]


class RandomExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(RandomExplainer, self).__init__(model, training_data, seed)

    def explain(self, instance, budget):
        # Not generating random importances for now, just setting all to 0
        features = np.random.choice(self.feature_names, budget)
        return [(feat, 0.0) for feat in features]


class LogisticExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(LogisticExplainer, self).__init__(model, training_data, seed)

    def explain(self, instance, budget, p=False):
        # Pair feature names with importances and sort
        instance = self.preprocessor.transform([instance])
        coef = self.model.coef_[0]
        importances = np.multiply(coef, instance)[0]

        pairs = sorted(
            zip(self.feature_names, importances),
            key = lambda x : abs(x[1]),
            reverse = True
        )
        if p: print(pairs[:10])
        return pairs[:budget]


class TreeExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(TreeExplainer, self).__init__(model, training_data, seed)

    def explain(self, instance, budget, p=False):
        # Pair feature names with importances and sort
        instance = self.preprocessor.transform([instance])[0]
        coef = self.model.feature_importances_
        importances = np.multiply(coef, instance)
        pairs = sorted(
            zip(self.feature_names, importances),
            key = lambda x : abs(x[1]),
            reverse = True
        )
        if p: print(pairs[:10])
        return pairs[:budget]


# TEXT EXPLAINERS ##############################################################

class GreedyTextExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(GreedyTextExplainer, self).__init__(
                model,
                training_data,
                seed,
                text=True)
        self.mask = '-----'

    def explain(self, instance, budget):
        vectorizer = CountVectorizer()
        preprocess = vectorizer.build_preprocessor()
        tokenize = vectorizer.build_tokenizer()
        tokens = tokenize(preprocess(instance))
        base = self.model.predict_proba([instance])[0, 1]
        importances = []
        for i in range(len(tokens)):
            hidden_word = tokens[i]
            tokens[i] = self.mask
            pred = self.model.predict_proba([' '.join(tokens)])[0, 1]
            importances.append( (hidden_word.lower(), base - pred) )
            tokens[i] = hidden_word

        feature_pairs = sorted(
            importances,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, importance in feature_pairs[:budget]:
            print('{0:10s}: {1:6.4f}'.format(feature, importance))
        print('------------')

        feats, _ = zip(*feature_pairs[:budget])
        return feats


class LimeTextExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(LimeTextExplainer, self).__init__(model, training_data, seed,
                text=True)
        self.explainer = LimeText(random_state=self.seed)

    def explain(self, instance, budget):
        exp = self.explainer.explain_instance(
                text_instance=instance,
                classifier_fn=self.model.predict_proba,
                num_features=budget)

        feature_pairs = exp.as_list()
        feats = []
        for feature, importance in feature_pairs:
            print('{0:10s}: {1:6.4f}'.format(feature, importance))
            feats.append(feature)
        print('------------')
        return feats


class ShapTextExplainer(Explainer):
    def __init__(self, model, training_data, seed):
        super(ShapTextExplainer, self).__init__(model, training_data, seed,
                text=True)
        self.preprocessor = Pipeline(steps=model.steps[:-1])
        model = model.steps[-1][1].module_
        background_data = self.preprocessor.transform(training_data[:100])
        background_data = torch.Tensor(background_data).long().to('cuda')
        self.explainer = DeepExplainer(model, background_data)

    def explain(self, instance, budget):
        print(instance)
        instance = self.preprocessor.transform([instance])
        print(instance)
        instance = torch.Tensor(instance).long().to('cuda')
        print(instance)
        values = self.explainer.shap_values(X=instance)
        print(values)
        exit()
        feature_pairs = sorted(
                zip(self.feature_names, values),
                key=lambda x : abs(x[1]),
                reverse=True
        )

        for feature, importance in feature_pairs:
            print('{0:10s}: {1:6.4f}'.format(feature, importance))
        print('------------')
        feats, _ = zip(*feature_pairs[:budget])
        return feats
