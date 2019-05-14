import os
import gzip
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def to_csv(in_path, out_path):
    gz = gzip.open(in_path, 'r')
    with open(out_path, 'w') as csv:
        for l in tqdm(gz):
            line = eval(l)
            text = str(line['reviewText'])
            rating = str(line['overall'])
            csv.write(rating + "," + text + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=('Convert an Amazon review dataset from .json.gz to a'
                         ' more managable csv.\nDatasets available'
                         ' at: http://jmcauley.ucsd.edu/data/amazon/'))
    parser.add_argument('i', metavar='path', help='Path of .json.gz file')
    parser.add_argument('-o', metavar='out_file',
                        help='Path to save .csv (default: <base name>.csv)')
    args = parser.parse_args()
    if args.o is None:
        args.o = args.i.split('.')[0] + '.csv'
    to_csv(args.i, args.o)
