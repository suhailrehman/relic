import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import glob
import os
import random


def generate_random_index_sample(num_rows, frac=0.2):
    num_samples = max(int(num_rows * frac), 1)
    return np.random.choice(range(num_rows + 1), num_samples, replace=False)


def sample_df(df, frac=0.2, sample_index=None):
    if not sample_index:
        sample_index = generate_random_index_sample(df, frac=frac)
    return df.loc[df.index.isin(sample_index)]


def skip_row_function(i, sample_index=None, frac=0.2):
    if i == 0:
        return False

    if sample_index is not None:
        return i-1 not in sample_index
    else:
        return i > 0 and random.random() > frac


def load_df_sample(filename, frac=0.2, sample_index=None, min_rows=1):
    full_size = get_file_rowcount(filename)
    if full_size < min_rows:
        return pd.read_csv(filename, index_col=0, header=0)
    else:
        return pd.read_csv(filename, index_col=0, header=0,
                           skiprows=lambda i: skip_row_function(i, sample_index=sample_index, frac=frac))


def build_sample_df_dict_dir(csv_dir, frac=0.2, sample_index=None):
    dataset = {}
    for file in tqdm(glob.glob(csv_dir + '*.csv')):
        csvfile = os.path.basename(file)
        dataset[csvfile] = load_df_sample(file, frac=frac, sample_index=sample_index)
    return dataset


def get_file_rowcount(filename, use_sampling=False):
    # Sample length technique http://www.documentroot.com/2011/02/approximate-line-count-for-very-large.html
    with open(filename, 'r') as fp:
        num_lines = sum(1 for line in fp)
    return num_lines


def generate_sample_index(csv_dir, output_dir, frac=0.2):
    max_row_count = max(get_file_rowcount(file) for file in tqdm(glob.glob(csv_dir + '*.csv')))
    index_sample = generate_random_index_sample(max_row_count, frac=frac)
    with open(output_dir + 'sample_index.txt', 'w') as fp:
        for i in index_sample:
            fp.write(str(i) + "\n")
    return index_sample
