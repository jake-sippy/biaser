import os
import gzip
import pickle
import numpy as np


def to_csv(path):
    g = gzip.open(path, 'r')
    save_filename = path.split('.json.gz')[0]
    with open(save_filename + '.csv', 'w') as f:
        for l in g:
            line = eval(l)
            text = str(line['reviewText'])
            rating = str(line['overall'])
            f.write(rating + "," + text + "\n")

if __name__ == '__main__':
    data_dir = 'data'
    gz_filename = 'reviews_Video_Games_5.json.gz'
    to_csv(os.path.join(data_dir, gz_filename))
