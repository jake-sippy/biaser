#!/bin/bash

git clone https://github.com/jake-sippy/biaser.git
cd biaser

source $(conda info --base)/etc/profile.d/conda.sh
conda env create -f environment.yaml
conda activate bias

# python download_datasets.py
echo "python full_test.py 0 10"
python full_test.py $dataset 0 10 $1
