import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer as LimeText
from shap import KernelExplainer, DeepExplainer, kmeans


class Explainer:
    def __init__(self, model, training_data):
        # Actual model must be last step
        self.training_data = training_data
        self.preprocessor = Pipeline(steps=model.steps[:-1])
        self.model = model.steps[-1][1]
        if 'counts' in model.named_steps:
            self.feature_names = model.named_steps['counts'].get_feature_names()
        else:
            assert False, 'Current model has no way to get feature names'

    def explain(self, instance, budget):
        # Return a list of tuples: (feature, importance). Sorted in decreasing
        # order of importance
        raise NotImplementedError


class LimeExplainer(Explainer):
    def __init__(self, model, training_data):
        super(LimeExplainer, self).__init__(model, training_data)
        data = self.preprocessor.transform(self.training_data)
        self.explainer = LimeTabularExplainer(
                training_data=data.astype(float),
                feature_names=self.feature_names)

    def explain(self, instance, budget, num_samples=5000):
        instance = self.preprocessor.transform([instance])[0]
        exp = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=self.model.predict_proba,
                num_samples=num_samples,
                num_features=budget)

        feature_pairs = exp.as_map()[1]
        feats = []
        for feat_idx, importance in feature_pairs:
            feats.append( (self.feature_names[feat_idx], importance) )
        return feats[:budget]


class BaggedLimeExplainer(Explainer):
    def __init__(self, model, training_data, n_bags=3, reduced=False):
        super(BaggedLimeExplainer, self).__init__(model, training_data)
        data = self.preprocessor.transform(self.training_data)

        # Create multiple LIME instances
        self.reduced = reduced
        self.n_bags = n_bags
        self.explainers = []
        for i in range(n_bags):
            explainer = LimeTabularExplainer(
                    training_data=data.astype(float),
                    feature_names=self.feature_names)
            self.explainers.append(explainer)

    def explain(self, instance, budget):
        instance = self.preprocessor.transform([instance])[0]

        # 5000 is the default num_samples in LIME
        if self.reduced:
            num_samples = int(5000 / self.n_bags)
        else:
            num_samples = 5000

        features = []
        importances = []

        for explainer in self.explainers:
            exp = explainer.explain_instance(
                    data_row=instance,
                    predict_fn=self.model.predict_proba,
                    num_samples=num_samples,
                    num_features=budget)
            feature_pairs = exp.as_map()[1]

            for feat_idx, importance in feature_pairs:
                feature_name = self.feature_names[feat_idx]
                if feature_name in features:
                    idx = features.index(feature_name)
                    importances[idx] += importance
                else:
                    features.append(feature_name)
                    importances.append(importance)

        importances = [i / self.n_bags for i in importances]
        pairs = list(zip(features, importances))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:budget]


class ShapExplainer(Explainer):
    def __init__(self, model, training_data):
        super(ShapExplainer, self).__init__(model, training_data)
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


class BaggedShapExplainer(Explainer):
    def __init__(self, model, training_data, n_bags=3):
        super(BaggedShapExplainer, self).__init__(model, training_data)
        data = self.preprocessor.transform(self.training_data)

        # Create multiple LIME instances
        self.n_bags = n_bags
        self.explainers = []
        for i in range(n_bags):
            explainer = ShapExplainer(model, training_data)
            self.explainers.append(explainer)

    def explain(self, instance, budget):
        features = []
        importances = []
        for explainer in self.explainers:
            pairs = explainer.explain(instance, budget)
            for feature_name, importance in pairs:
                if feature_name in features:
                    idx = features.index(feature_name)
                    importances[idx] += importance
                else:
                    features.append(feature_name)
                    importances.append(importance)

        importances = [i / self.n_bags for i in importances]
        pairs = list(zip(features, importances))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        return pairs[:budget]



class GreedyExplainer(Explainer):
    def __init__(self, model, training_data):
        super(GreedyExplainer, self).__init__(model, training_data)

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
    def __init__(self, model, training_data):
        super(RandomExplainer, self).__init__(model, training_data)

    def explain(self, instance, budget):
        # Not generating random importances for now, just setting all to 0
        features = np.random.choice(self.feature_names, budget)
        return [(feat, 0.0) for feat in features]


class LogisticExplainer(Explainer):
    def __init__(self, model, training_data):
        super(LogisticExplainer, self).__init__(model, training_data)

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
    def __init__(self, model, training_data):
        super(TreeExplainer, self).__init__(model, training_data)

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
