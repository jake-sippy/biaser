import numpy as np
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer
from shap import KernelExplainer, kmeans


class Explainer:
    def __init__(self, model, training_data, feature_names, seed):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.seed = seed

    def explain(self, instance, budget):
        # Return a list of the most important features (by name) in decreasing
        # order
        raise NotImplementedError


class LimeExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(LimeExplainer, self).__init__(model, training_data,
                feature_names, seed)
        self.explainer = LimeTabularExplainer(
                training_data=self.training_data,
                feature_names=self.feature_names,
                random_state=self.seed)

    def explain(self, instance, budget):
        exp = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=self.model.predict_proba,
                num_features=budget)

        feature_pairs = exp.as_map()[1]
        feats = []
        for feat_idx, importance in feature_pairs:
            feats.append(self.feature_names[feat_idx])
        return feats


class ShapExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(ShapExplainer, self).__init__(model, training_data,
                feature_names, seed)
        background_data = kmeans(training_data, 100)
        self.explainer = KernelExplainer(
                model=self.model.predict_proba,
                data=background_data)

    def explain(self, instance, budget):
        values = self.explainer.shap_values(
                X=instance,
                nsamples="auto",
                l1_reg='num_features(%d)' % budget)[1]
        pairs = sorted(
                zip(self.feature_names, values),
                key=lambda x : abs(x[1]),
                reverse=True
        )
        feats, _ = zip(*pairs[:budget])
        return feats


class ShapZerosExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(ShapZerosExplainer, self).__init__(model, training_data,
                feature_names, seed)
        background_data = np.zeros((1, len(feature_names)))
        self.explainer = KernelExplainer(
                model=self.model.predict_proba,
                data=background_data)

    def explain(self, instance, budget):
        values = self.explainer.shap_values(
                X=instance,
                nsamples="auto",
                l1_reg='num_features(%d)' % budget)[1]
        pairs = sorted(
                zip(self.feature_names, values),
                key=lambda x : abs(x[1]),
                reverse=True
        )
        feats, _ = zip(*pairs[:budget])
        return feats


class ShapMedianExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(ShapMedianExplainer, self).__init__(model, training_data,
                feature_names, seed)
        background_data = np.median(training_data, axis=0).reshape(1, -1)
        self.explainer = KernelExplainer(
                model=self.model.predict_proba,
                data=background_data)

    def explain(self, instance, budget):
        values = self.explainer.shap_values(
                X=instance,
                nsamples="auto",
                l1_reg='num_features(%d)' % budget)[1]
        pairs = sorted(
                zip(self.feature_names, values),
                key=lambda x : abs(x[1]),
                reverse=True
        )
        feats, _ = zip(*pairs[:budget])
        return feats


class GreedyExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(GreedyExplainer, self).__init__(model, training_data,
                feature_names, seed)

    def explain(self, instance, budget):
        inst = instance.copy().reshape(1, -1)
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
        feats, _ = zip(*pairs[:budget])
        return feats

class RandomExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(RandomExplainer, self).__init__(model, training_data,
                feature_names, seed)

    def explain(self, instance, budget):
        return np.random.choice(self.feature_names, budget)


class LogisticExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(LogisticExplainer, self).__init__(model, training_data,
                feature_names, seed)

    def explain(self, instance, budget, p=False):
        # Pair feature names with importances and sort
        coef = self.model.coef_[0]
        importances = np.multiply(coef, instance)

        pairs = sorted(
            zip(self.feature_names, importances),
            key = lambda x : abs(x[1]),
            reverse = True
        )
        if p: print(pairs[:10])

        # Return top feature names
        feats, _ = zip(*pairs[:budget])
        return feats


class TreeExplainer(Explainer):
    def __init__(self, model, training_data, feature_names, seed):
        super(DecisionTreeExplainer, self).__init__(model, training_data,
                feature_names, seed)

    def explain(self, instance, budget, p=False):
        # Pair feature names with importances and sort
        coef = self.model.feature_importances_
        importances = np.multiply(coef, instance)
        pairs = sorted(
            zip(self.feature_names, importances),
            key = lambda x : abs(x[1]),
            reverse = True
        )
        if p: print(pairs[:10])

        # Return top feature names
        feats, _ = zip(*pairs[:budget])
        return feats
