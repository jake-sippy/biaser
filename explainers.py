import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer as LimeText
from shap import KernelExplainer, DeepExplainer, kmeans

# Image-only
import cv2
from torch.autograd import Variable
from lime.lime_image import LimeImageExplainer as LimeImage
from grad_cam import GradCAM

import gradients


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



# IMAGE EXPLAINERS #############################################################

class ImageExplainer:
    def __init__(self):
        pass

    def explain(self, instance, budget):
        pass


class LimeImageExplainer(ImageExplainer):
    def __init__(self, model, label):
        self.explainer = LimeImage()
        self.model = model
        self.label = label

    def explain(self, instance, budget):

        def classifier_fn(instances):
            instances = np.moveaxis(instances, -1, 1)
            instances = torch.tensor(instances).cuda().float()
            with torch.no_grad():
                output = self.model.predict_proba(instances)
            return output.cpu().numpy()

        instance = instance.double().detach().cpu().numpy()
        instance = np.moveaxis(instance, 0, -1)

        # Heuristic: each super pixel is about 5% of total pixels
        # max ensures we return at least one superpixel
        num_features = max(budget // 5, 1)

        exp = self.explainer.explain_instance(instance, classifier_fn)
        _, mask = exp.get_image_and_mask(
                label=self.label,
                positive_only=True,
                # negative_only=False,
                hide_rest=False,
                num_features=num_features,
                min_weight=0)

        mask = mask.astype(np.float32)

        # # only return above percentile
        # top_percentile = np.percentile(mask, 100 - budget)
        # mask[ mask < top_percentile ] = 0.0
        # mask[ mask >= top_percentile ] = 1.0

        return mask


class SmoothGradExplainer(ImageExplainer):

    def __init__(self, model, label, cuda=True):
        self.explainer = gradients.SmoothGrad(
            pretrained_model=model.model,
            cuda=cuda,
            stdev_spread=0.15,
            n_samples=25,
            magnitude=True
        )
        self.model = model
        self.label = label

    def explain(self, instance, budget):
        # Add necessary preprocessing (batch dim, variable wrapper)
        instance = instance.unsqueeze(0)
        instance = Variable(instance.cuda(), requires_grad=True)
        explanation = self.explainer(instance) #, index=self.label)

        # grad explainers return in 3-d
        explanation_2d = np.sum(explanation, axis=0)

        # only return above percentile
        top_percentile = np.percentile(explanation_2d, 100 - budget)
        explanation_2d[ explanation_2d < top_percentile ] = 0.0
        explanation_2d[ explanation_2d >= top_percentile ] = 1.0

        return explanation_2d


class VanillaGradExplainer(ImageExplainer):

    def __init__(self, model, label, cuda=True):
        self.explainer = gradients.VanillaGrad(
            pretrained_model=model.model,
            cuda=cuda,
        )
        self.model = model
        self.label = label

    def explain(self, instance, budget):
        # Add necessary preprocessing (batch dim, variable wrapper)
        instance = instance.unsqueeze(0)
        instance = Variable(instance.cuda(), requires_grad=True)
        explanation = self.explainer(instance) #, index=self.label)

        # grad explainers return in 3-d
        explanation_2d = np.sum(np.abs(explanation), axis=0)
        top_percentile = np.percentile(explanation_2d, 100 - budget)

        # only return above percentile
        explanation_2d[ explanation_2d < top_percentile ] = 0.0
        explanation_2d[ explanation_2d >= top_percentile ] = 1.0

        return explanation_2d

class RandomImageExplainer(ImageExplainer):
    def explain(self, instance, budget):
        c, w, h = instance.shape
        area = w * h
        mask = np.zeros(area, dtype=np.float32)
        num_pixels = round( (budget / 100.0) * area )
        chosen = np.random.choice(np.arange(area), size=num_pixels ,replace=False)
        mask[chosen] = 1.0
        return mask.reshape(w, h)


class ShapImageExplanier(ImageExplainer):
    def __init__(self, model, label, dataset):
        images = []
        length = len(dataset)
        idxs = np.random.choice(length, min(100, length), replace=False)
        for i in idxs:
            images.append(dataset[i]['image'])

        background_data = torch.stack(images)
        if torch.cuda.is_available():
            background_data = background_data.cuda()

        self.label = label
        self.model = model
        self.explainer = DeepExplainer(self.model.model, background_data[:25])

    def explain(self, instance, budget):
        instance = instance.unsqueeze(0)
        _, c, w, h = instance.shape
        shap_values = self.explainer.shap_values(instance)[self.label][0]
        # print(shap_values)
        # print(shap_values.shape)

        explanation_2d = np.sum(np.abs(shap_values), axis=0)
        top_percentile = np.percentile(explanation_2d, 100 - budget)

        # only return above percentile
        explanation_2d[ explanation_2d < top_percentile ] = 0.0
        explanation_2d[ explanation_2d >= top_percentile ] = 1.0

        return explanation_2d.astype(np.float32)


class GradCamExplainer(ImageExplainer):
    def __init__(self, model_wrapper, target_layer, label):
        self.model = model_wrapper.model
        self.explainer = GradCAM(model=self.model)
        self.target_layer = target_layer
        self.label = label

    def explain(self, instance, budget):
        instance = instance.unsqueeze(0).cuda()

        probs, ids = self.explainer.forward(instance)
        # print(probs, ids)
        target_ids = torch.LongTensor([[self.label]]).cuda()
        self.explainer.backward(ids=target_ids)
        regions = self.explainer.generate(target_layer=self.target_layer)
        explanation_3d = regions[0].double().detach().cpu().numpy()

        explanation_2d = np.sum(np.abs(explanation_3d), axis=0)
        top_percentile = np.percentile(explanation_2d, 100 - budget)

        # only return above percentile
        explanation_2d[ explanation_2d < top_percentile ] = 0.0
        explanation_2d[ explanation_2d >= top_percentile ] = 1.0

        return explanation_2d.astype(np.float32)

