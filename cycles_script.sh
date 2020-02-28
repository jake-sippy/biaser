#!/bin/bash

conda env create -f environment.yaml
conda activate bias
python download_datasets.py
for dataset in datasets/*; do
    echo "python full_test.py $dataset 0 10"
    python full_test.py $dataset 0 10
done
