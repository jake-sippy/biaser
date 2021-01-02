# This script is meant to implement ROAR in pytorch, as well as run and time it
# for comparison against data-staining on the warbler vs. sparrow bird
# classification dataset
import json
import os
import time

from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from explainers import SmoothGradExplainer, VanillaGradExplainer
from models import PretrainedModels

# from models import PretrainedModels

INPUT_SIZE = 224
BATCH_SIZE = 32
NORMALIZE_MEANS = [0.485, 0.456, 0.406]
NORMALIZE_STDS = [0.229, 0.224, 0.225]

TRANSFORMS = {
    "train": A.Compose(
        [
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.CenterCrop(INPUT_SIZE, INPUT_SIZE),
            A.HorizontalFlip(),
            A.Normalize(NORMALIZE_MEANS, NORMALIZE_STDS),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    ),
    "val": A.Compose(
        [
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.CenterCrop(INPUT_SIZE, INPUT_SIZE),
            A.Normalize(NORMALIZE_MEANS, NORMALIZE_STDS),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy"),
    ),
}


def main():
    # train_dataset = BirdDataset("train", TRANSFORMS["train"])
    # val_dataset = BirdDataset("val", TRANSFORMS["val"])
    # test_dataset = BirdDataset("test", TRANSFORMS["val"])
    time_start = time.time()

    runlog = {}

    seeds = list(range(1))

    explainers = {
        "smooth": SmoothGradExplainer,
        "vanilla": VanillaGradExplainer,
    }

    dataloaders_dict = {
        split: torch.utils.data.DataLoader(
            RoarBirdDataset(split, explainers),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        for split in ["train", "val", "test"]
    }

    ## Fit #seeds models to the original dataset, then create new datasets by
    ## explaining every example w/ every explainer and varying % pixels replaced
    ## with the mean

    for seed in seeds:
        runlog["seed"] = seed

        path = "roar/seed_{:02d}/".format(seed)
        model_path = path + "orig_model.pt"
        orig_model = PretrainedModels(2, "resnet50")
        if os.path.exists(model_path):
            print("Loading model, Seed {:02d}: {:s}".format(seed, model_path))
            orig_model.load(model_path)
        else:
            orig_model.fit(dataloaders_dict, runlog)
            print("Saving model, Seed {:02d}: {:s}".format(seed, model_path))
            orig_model.save(model_path, -1)

        orig_model.eval(dataloaders_dict, "test")
        orig_f1 = orig_model.curr_f1

        runlog["model_path"] = model_path
        runlog["results"] = {}

        for split in dataloaders_dict:
            dataloaders_dict[split].dataset.set_base_model(orig_model)

        for explainer in explainers:
            results = {}

            for split in dataloaders_dict:
                dataloaders_dict[split].dataset.set_curr_explainer(explainer)

            xs = [0]
            ys = [orig_f1]

            # [5, 10, 15, ..., 100]
            for t in range(5, 101, 5):
                test_model = PretrainedModels(2, "resnet50")
                for split in dataloaders_dict:
                    dataloaders_dict[split].dataset.set_percentage_masked(t)
                test_model.fit(dataloaders_dict, runlog)
                test_model.eval(dataloaders_dict, "test")
                xs.append(t)
                ys.append(test_model.curr_f1)

            results["percentages"] = xs
            results["f1_scores"] = ys
            results["time_elapsed"] = time.time() - time_start

            runlog["results"][explainer] = results

            plt.plot(xs, ys)
            plt.show()

        runlog["time_elapsed"] = time.time() - time_start

        log_path = path + "log.json"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(log_path, "w") as f:
            json.dump(runlog, f, indent=4)


################################################################################
# Shamelessly hacky copy from image_utils to work with the bird dataset, but for
# ROAR comparison

NUM_ATTRS = 312
CUB_DATASET_LOCATION = "datasets/CUB_200_2011/"


class RoarBirdDataset(Dataset):
    def __init__(
        self,
        split,
        explainers,  # Dict of explainer names to explainers
        binary_classes=["Warbler", "Sparrow"],
    ):

        assert split in ["train", "val", "test", "full"], (
            "Unknown split passed to BirdDataset (%s)" % split
        )

        self.data_dir = CUB_DATASET_LOCATION
        self.mode = split
        self.transform = TRANSFORMS["train" if split == "train" else "val"]

        # Build self.data DataFrame ###########################################

        # img_id -> img_path
        data = pd.read_csv(
            self.data_dir + "images.txt",
            sep=" ",
            header=None,
            names=["img_id", "img_path"],
        )

        # Manually drop malformed examples by id number:
        #   - 5007 (user clicks are outside img dimensions)
        data = data[data["img_id"] != 5007]

        # Binarize data #######################################################
        assert len(binary_classes) == 2
        bird_labels = {binary_classes[0]: 0, binary_classes[1]: 1}

        def get_name(row):
            return row["img_path"].split("_")[-3]

        data["bird_name"] = data.apply(get_name, axis=1)
        data["label"] = data["bird_name"].map(bird_labels)
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Train / Val / Test Split ############################################
        if split != "full":
            train_test_data = pd.read_csv(
                self.data_dir + "train_test_split.txt",
                sep=" ",
                header=None,
                names=["img_id", "is_train"],
            )

            # 80/10/10 split
            is_train = data.merge(train_test_data, on="img_id")["is_train"] == 1

            if split == "train":
                data = data[is_train]
                data, _ = train_test_split(
                    data, train_size=0.8888, random_state=1
                )
            elif split == "val":
                data = data[is_train]
                _, data = train_test_split(
                    data, train_size=0.8888, random_state=1
                )
            elif split == "test":
                data = data[~is_train]

        data.reset_index(drop=True, inplace=True)
        self.data = data
        self.explainers = explainers

        self.percentage_masked = 0
        self.curr_explainer = None
        self.base_model = None

        # CACHE explainer_name -> (id -> explanation)
        self.cache = {}
        for explainer_name in explainers:
            self.cache[explainer_name] = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        img_id = row["img_id"]
        label = row["label"].astype(int)
        path = row["img_path"]

        with open(self.data_dir + "images/" + path, "rb") as f:
            image_file = Image.open(f)
            image_rgb = image_file.convert("RGB")
            image_rgb = np.array(image_rgb)

        transformed = self.transform(image=image_rgb, keypoints=[])
        image = transformed["image"]

        # Handle explanation masking
        if self.curr_explainer is not None:
            if img_id in self.cache[self.curr_explainer]:
                # print("cache hit")
                explanation = self.cache[self.curr_explainer][img_id]
            else:
                assert self.base_model is not None
                # always use label 1 and 100% budget for explainer
                # print("cache miss")
                exp = self.explainers[self.curr_explainer](self.base_model, 1)
                explanation = exp.explain(
                    image, self.percentage_masked, mask=False
                )
                self.cache[self.curr_explainer][img_id] = explanation

            percentile = 100 - self.percentage_masked
            top_percentile = np.percentile(explanation, percentile)
            explanation[explanation < top_percentile] = False
            explanation[explanation >= top_percentile] = True
            explanation = torch.from_numpy(explanation).bool()

            # DEBUG
            # plt.title("explanation")
            # plt.imshow(explanation)
            # plt.show()
            #
            # plt.title("image")
            # plt.imshow(image.permute(1, 2, 0))
            # plt.show()

            avg = torch.mean(image, [1, 2])
            for pixel in explanation.nonzero():
                image[:, pixel[0], pixel[1]] = avg

            # DEBUG
            # plt.title("masked image")
            # plt.imshow(image.permute(1, 2, 0))
            # plt.show()

        return {
            "img_id": img_id,
            "image": image,
            # "attrs": attrs,
            "label": label,
            "path": path,
        }

    def set_percentage_masked(self, perc):
        assert 0 <= perc <= 100, "invalid percentage"
        self.percentage_masked = perc

    def set_curr_explainer(self, explainer_name):
        valid_explainers = list(self.explainers.keys()) + [None]
        assert explainer_name in valid_explainers
        self.curr_explainer = explainer_name

    def set_base_model(self, base_model):
        self.base_model = base_model


if __name__ == "__main__":
    main()
