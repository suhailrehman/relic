import os

import pandas as pd
import pytest
import logging
import glob
from itertools import product
from time import perf_counter_ns, perf_counter

from relic.utils.serialize import build_df_dict_dir
from relic.approx.sampling import load_df_sample, get_file_rowcount, generate_sample_index

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/benchmark/')
files = [os.path.basename(f) for f in glob.glob(data_dir+'*.csv')]
logging.info(f"Benchmark Files: {files}")
sample_sizes = [0.05, 0.1, 0.25, 0.5, 0.75]
counter = perf_counter


@pytest.mark.parametrize("file", files)
def test_benchmark_read_csv(file):
    logging.info(f'Benchmarking Full CSV read of file: {file}')
    start = counter()
    df = pd.read_csv(data_dir+file)
    end = counter()
    logging.info(f'Total Loading time: {end-start}')
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize("file, frac", product(files, sample_sizes))
def test_benchmark_read_random_sample(file, frac):
    logging.info(f'Benchmarking {frac}=-sampled CSV Read of File: {file}')
    start = counter()
    sampled_df = load_df_sample(data_dir+'/'+file, frac=frac)
    end = counter()
    logging.info(f'Total Loading time: {end-start}')
    assert isinstance(sampled_df, pd.DataFrame)


@pytest.mark.parametrize("file, frac", product(files, sample_sizes))
def test_benchmark_read_indexed_sample(tmpdir_factory, file, frac):
    logging.info(f'Benchmarking {frac}=-sampled CSV Read of File: {file}')
    output_dir = tmpdir_factory.mktemp(f'inferred_sample_{frac}')
    start = counter()
    sample_index = generate_sample_index(data_dir, output_dir, frac=frac)
    end = counter()
    logging.info(f'Index Sample Generation Time: {end-start}')
    start = counter()
    sampled_df = load_df_sample(data_dir+'/'+file, frac=frac, sample_index=sample_index)
    end = counter()
    logging.info(f'Sample Loading time: {end-start}')
    assert isinstance(sampled_df, pd.DataFrame)