#!/bin/bash
# G: will install dependencies for cuda 10.1; No guarantees for other versions 
set -x

# LIME depends on pillow, which requires 3.6
conda install python=3.6

# install AllenNLP from source, required by roberta models
git clone https://github.com/allenai/allennlp.git
cd allennlp
# HEAD is buggy, so use this commit
git checkout 236e1fd01ca30409cd736625901292609009f5c4
pip install --editable .
pip install -r dev-requirements.txt
cd ..

# install cuda toolkit, for some reason allennlp did not install it
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# install other dependencies
pip install lime 
