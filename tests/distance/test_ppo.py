import os
import pandas as pd
from relic.distance.ppo import *
import pytest
import logging

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, '../data/test_workflow/artifacts/')
file1 = '0.csv'
file2 = '1.csv'


def test_compute_all_ppo():
    df1 = pd.read_csv(data_dir+file1, index_col=0)
    df2 = pd.read_csv(data_dir+file2, index_col=0)
    df_dict = {file1: df1, file2: df2}
    result_dict = compute_all_ppo_labels(file1, file2, df_dict)[1]
    logging.debug(f'Result Dict: {result_dict}')
    assert result_dict['jaccard'] == pytest.approx(0.999, rel=1e-3)
    assert result_dict['containment'] == pytest.approx(0.999, rel=1e-3)
    assert result_dict['overlap'] == pytest.approx(20999.0, rel=1e-3)
    assert result_dict['containment_oneside'] == pytest.approx(0.999, rel=1e-3)


def test_compute_baseline():
    df1 = pd.read_csv(data_dir+file1, index_col=0)
    df2 = pd.read_csv(data_dir+file2, index_col=0)
    df_dict = {file1: df1, file2: df2}
    result_dict = compute_baseline(file1, file2, df_dict)
    logging.debug(f'Result Dict: {result_dict}')
    assert 'baseline' in result_dict.keys()
    assert 0.0 <= result_dict['baseline'] <= 1.0
