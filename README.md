### Data Staining

To install all required dependencies run
`./setup.sh`

The CUB_200_2011 dataset must be manually downloaded and extracted from 
https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view

Extract `CUB_200_2011.tgz` into the `datasets/` folder created by the setup
script making sure to move `attributes.txt` into `CUB_200_2011/`

    cd datasets/
    tar -xzfv CUB_200_2011.tgz
    mv attributes.txt CUB_200_2011/
