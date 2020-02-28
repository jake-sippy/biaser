import os
import io
import gzip
import tarfile
import argparse
import pandas as pd
from urllib.request import urlopen
from sklearn.datasets import fetch_20newsgroups

DATASET_DIR = 'datasets'

# Load the IMDb Reviews dataset
def load_imdb(path):
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    response = urlopen(url)
    memfile = io.BytesIO(response.read())
    tar = tarfile.open(fileobj=memfile, mode="r:gz")
    reviews = []
    labels = []
    for member in tar.getmembers():
        if member.isfile():
            fname = member.name.split('/')[-1]
            if fname.endswith('.txt') and fname[0].isdigit():
                rating = int(fname.split('.txt')[0].split('_')[-1])
                if rating > 0:
                    reviews.append(tar.extractfile(member).read())
                    labels.append(1 if rating > 5 else 0)

    df = pd.DataFrame(data=[i for i in zip(reviews, labels)])
    filename = os.path.join(path, 'imdb.csv')
    df.to_csv(filename, index=False, header=False)


# Load the Amazon Cell Phones and Accessories Reviews dataset
def load_amazon_cell(path):
    print('Loading the Amazon Cell Phones and Accessories Reviews dataset...')
    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/' \
          'reviews_Cell_Phones_and_Accessories_5.json.gz'
    response = urlopen(url)
    data = response.read()
    data = gzip.decompress(data)
    dataset = pd.read_json(data, lines=True)
    dataset = dataset[dataset['overall'] != 3]
    dataset['overall'] = dataset['overall'].map(
            {1:0, 2:0, 4:1, 5:1}
    )
    filename = os.path.join(path, 'amazon_cell.csv')
    dataset.to_csv(filename, index=False, header=False,
            columns=['reviewText', 'overall'])


# Load the Amazon Home and Kitchen Reviews dataset
def load_amazon_home(path):
    print('Loading the Amazon Home and Kitchen Reviews dataset...')
    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/' \
          'reviews_Home_and_Kitchen_5.json.gz'
    response = urlopen(url)
    data = response.read()
    data = gzip.decompress(data)
    dataset = pd.read_json(data, lines=True)
    dataset = dataset[dataset['overall'] != 3]
    dataset['overall'] = dataset['overall'].map(
            {1:0, 2:0, 4:1, 5:1}
    )
    filename = os.path.join(path, 'amazon_home.csv')
    dataset.to_csv(filename, index=False, header=False,
            columns=['reviewText', 'overall'])

# Load the 20 Newsgroups dataset Atheism v. Christianity
def load_newsgroups_atheism(path):
    data = fetch_20newsgroups(
            remove=('headers', 'footers', 'quotes'),
            categories=['alt.atheism', 'soc.religion.christian']
    )
    df = pd.DataFrame(data=zip(data.data, data.target))
    filename = os.path.join(path, 'newsgroups_atheism.csv')
    df.to_csv(filename, index=False, header=False)


# Load the 20 Newsgroups dataset Baseball v. Hockey
def load_newsgroups_baseball(path):
    data = fetch_20newsgroups(
            remove=('headers', 'footers', 'quotes'),
            categories=['rec.sport.baseball', 'rec.sport.hockey']
    )
    df = pd.DataFrame(data=zip(data.data, data.target))
    filename = os.path.join(path, 'newsgroups_baseball.csv')
    df.to_csv(filename, index=False, header=False)


# Load the 20 Newsgroups dataset IBM v. Mac
def load_newsgroups_ibm(path):
    data = fetch_20newsgroups(
            remove=('headers', 'footers', 'quotes'),
            categories=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
    )
    df = pd.DataFrame(data=zip(data.data, data.target))
    filename = os.path.join(path, 'newsgroups_ibm.csv')
    df.to_csv(filename, index=False, header=False)


if __name__ == '__main__':
    directory = DATASET_DIR
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Load datasets
    load_imdb(directory)
    load_amazon_cell(directory)
    load_amazon_home(directory)
    load_newsgroups_atheism(directory)
    load_newsgroups_baseball(directory)
    load_newsgroups_ibm(directory)
