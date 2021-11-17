import os
import pytest
import logging
import glob
import pandas as pd

from relic.distance.nppo import sample_groupby_detector
from relic.approx.sampling import load_df_sample, get_file_rowcount, generate_sample_index, build_sample_df_dict_dir

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(THIS_DIR, '../data/test_workflow/artifacts/')
files = [os.path.basename(f) for f in glob.glob(test_data_dir+'*.csv')]

syn_data_dir = '/tank/local/suhail/data/relic/relic_datasets_vldb_2021/dataset_flat_exact_sample_timing/20210126-153905/artifacts/'


@pytest.mark.parametrize("file1, file2, data_dir", (['5.csv', '6.csv', test_data_dir],
                                                    ['15.csv', '12.csv', syn_data_dir]))
def test_sample_groupby_detector(file1, file2, data_dir):
    src = pd.read_csv(data_dir+file1, index_col=0)
    dst = pd.read_csv(data_dir+file2, index_col=0)
    df_dict = {file1: src, file2: dst}
    sampled_df_dict = build_sample_df_dict_dir(data_dir, frac=1.00)
    result_tuple = sample_groupby_detector(file1, file2, df_dict, sampled_df_dict, 0.10)

    logging.info(f'GBD Test Result: {result_tuple}')
    assert result_tuple[1]['groupby'] >= 1.0
