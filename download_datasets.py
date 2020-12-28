import os
import io
import gzip
import zipfile
import tarfile
import argparse
import pandas as pd
import numpy as np
from urllib.request import urlopen
import requests
from sklearn.datasets import fetch_20newsgroups

DATASET_DIR = 'datasets'


def load_goodreads(path):
    print('goodreads...', end='')
    filename = os.path.join(path, 'goodreads.csv')
    if os.path.exists(filename):
        print('already exists')
        return

    file_id = '1908GDMdrhDN7sTaI_FelSHxbwcNM1EzR'
    tmp_filename = 'tmp_goodreads.json.gz'
    _download_file_from_google_drive(file_id, tmp_filename)
    with open(tmp_filename, 'rb') as f:
        gzip_file = f.read()
    data = gzip.decompress(gzip_file)

    dataset = pd.read_json(data, lines=True)

    keep_ratings = [1, 2, 4, 5]
    dataset = dataset[dataset['rating'].isin(keep_ratings)]
    dataset['rating'] = dataset['rating'].map( {1:0, 2:0, 4:1, 5:1} ).astype(int)

    # drop empty string reviews
    dataset.replace('', np.nan, inplace=True)
    dataset.dropna(axis=0, inplace=True)

    # write file
    dataset.to_csv(filename, index=False, header=False,
            columns=['review_text', 'rating'])
    os.remove(tmp_filename)
    print('done')


# Load the IMDb Reviews dataset
def load_imdb(path):
    print('imdb...', end='')
    filename = os.path.join(path, 'imdb.csv')
    if os.path.exists(filename):
        print('already exists')
        return

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
    df.to_csv(filename, index=False, header=False)
    print('done')


# Load the Amazon Cell Phones and Accessories Reviews dataset
def load_amazon_cell(path):
    print('amazon_cell...', end='')
    filename = os.path.join(path, 'amazon_cell.csv')
    if os.path.exists(filename):
        print('already exists')
        return

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

    dataset.replace('', np.nan, inplace=True)
    dataset.dropna(axis=0, inplace=True)

    dataset.to_csv(filename, index=False, header=False,
            columns=['reviewText', 'overall'])
    print('done')


# Load the Amazon Home and Kitchen Reviews dataset
def load_amazon_home(path):
    print('amazon_home...', end='')
    filename = os.path.join(path, 'amazon_home.csv')
    if os.path.exists(filename):
        print('already exists')
        return

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
    dataset.to_csv(filename, index=False, header=False,
            columns=['reviewText', 'overall'])
    print('done')


# Load the 20 Newsgroups dataset Atheism v. Christianity
def load_newsgroups_atheism(path):
    print('newsgroups_atheism...', end='')
    filename = os.path.join(path, 'newsgroups_atheism.csv')
    if os.path.exists(filename):
        print('already exists')
        return

    data = fetch_20newsgroups(
            remove=('headers', 'footers', 'quotes'),
            categories=['alt.atheism', 'soc.religion.christian']
    )
    df = pd.DataFrame(data=zip(data.data, data.target))
    df.to_csv(filename, index=False, header=False)
    print('done')


# Load the 20 Newsgroups dataset Baseball v. Hockey
def load_newsgroups_baseball(path):
    print('newsgroups_baseball...', end='')
    filename = os.path.join(path, 'newsgroups_baseball.csv')
    if os.path.exists(filename):
        print('already exists')
        return

    data = fetch_20newsgroups(
            remove=('headers', 'footers', 'quotes'),
            categories=['rec.sport.baseball', 'rec.sport.hockey']
    )
    df = pd.DataFrame(data=zip(data.data, data.target))
    df.to_csv(filename, index=False, header=False)
    print('done')


# Load the 20 Newsgroups dataset IBM v. Mac
def load_newsgroups_ibm(path):
    print('newsgroups_ibm...', end='')
    filename = os.path.join(path, 'newsgroups_ibm.csv')
    if os.path.exists(filename):
        print('already exists')
        return

    data = fetch_20newsgroups(
            remove=('headers', 'footers', 'quotes'),
            categories=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
    )
    df = pd.DataFrame(data=zip(data.data, data.target))
    df.to_csv(filename, index=False, header=False)
    print('done')


# HELPERS ######################################################################

def _download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    CHUNK_SIZE = 32768
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)

    # get confirm token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    directory = DATASET_DIR
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Load datasets
    print('Downloading datasets')
    # load_imdb(directory)
    # load_amazon_cell(directory)
    load_goodreads(directory)

    # Old Datasets
    # load_amazon_home(directory)
    # load_newsgroups_atheism(directory)
    # load_newsgroups_baseball(directory)
    # load_newsgroups_ibm(directory)
