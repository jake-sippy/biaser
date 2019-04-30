import os
import numpy as np

split_ratio = 0.8

def split_csv(base_filename, train_filename, test_filename):
    with open(base_filename, 'r') as f:
        print('Opened base file')
        lines = []
        for i, l in enumerate(f):
            lines.append(l)
        num_lines = i + 1
        idx = np.random.permutation(num_lines)
        split = int(num_lines * split_ratio)
        train_idx = idx[: split]
        test_idx = idx[split :]

        with open(train_filename, 'w') as train_f:
            print('Writing train split')
            for i in train_idx:
                train_f.write(lines[i])

        with open(test_filename, 'w') as test_f:
            print('Writing test split')
            for i in test_idx:
                test_f.write(lines[i])

if __name__ == '__main__':
    data_dir = 'data'
    csv_filename = 'reviews_Video_Games_5.csv'
    csv_path = os.path.join(data_dir, csv_filename)
    train_filename = csv_path.split('.csv')[0] + '_train.csv'
    test_filename  = csv_path.split('.csv')[0] + '_test.csv'
    split_csv(csv_path, train_filename, test_filename)
