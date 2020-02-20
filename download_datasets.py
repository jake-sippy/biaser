import os
import io
import gzip
import tarfile
import argparse
import pandas as pd
from urllib.request import urlopen
from sklearn.datasets import fetch_20newsgroups


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
def load_amazon(path):
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
    filename = os.path.join(path, 'amazon.csv')
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

# Load the 20 Newsgroups dataset
def load_newsgroups(path):
    data = fetch_20newsgroups(
            remove=('headers', 'footers', 'quotes'),
            categories=['alt.atheism', 'soc.religion.christian']
    )
    df = pd.DataFrame(data=[i for i in zip(data.data, data.target)])
    filename = os.path.join(path, 'newsgroups.csv')
    df.to_csv(filename, index=False, header=False)


if __name__ == '__main__':
    description = 'Load all datasets and processes them to be in a consistent' \
                  ' format and stores them in the specified directory.'
    parser = argparse.ArgumentParser(description=description)
    default_dir = 'datasets'
    parser.add_argument(
            '--dir',
            type=str,
            metavar='DIR',
            default=default_dir,
            help='Path to save datasets (default = {}/)'.format(default_dir)
    )
    args = parser.parse_args()

    directory = args.dir
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Load datasets
    # load_imdb(directory)
    # load_amazon(directory)
    load_amazon_home(directory)
    # load_newsgroups(directory)
