import os
import pandas as pd

from relic.distance.nppo import join_detector, groupby_detector, pivot_detector
# from relic.distance.ppo import *
import pytest
import logging


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, '../data/test_workflow/artifacts/')


def test_join_detector():
    file1 = '2.csv'
    file2 = '3.csv'
    file3 = '4.csv'
    src1 = pd.read_csv(data_dir+file1, index_col=0)
    src2 = pd.read_csv(data_dir+file2, index_col=0)
    dst = pd.read_csv(data_dir+file3, index_col=0)
    df_dict = {file1: src1, file2: src2, file3: dst}
    result_tuple = join_detector(file1, file2, file3, df_dict)
    logging.info(f'JD Test Result: {result_tuple}')
    assert result_tuple[1]['join'] == pytest.approx(1.0)
    
    
def test_groupby_detector():
    file1 = '5.csv'
    file2 = '6.csv'
    src = pd.read_csv(data_dir+file1, index_col=0)
    dst = pd.read_csv(data_dir+file2, index_col=0)
    df_dict = {file1: src, file2: dst}
    result_tuple = groupby_detector(file1, file2, df_dict)
    logging.info(f'GBD Test Result: {result_tuple}')
    assert result_tuple[1]['groupby'] >= 1.0


def test_pivot_detector():
    file1 = '6.csv'
    file2 = '7.csv'
    src = pd.read_csv(data_dir+file1, index_col=0)
    dst = pd.read_csv(data_dir+file2, index_col=0)
    df_dict = {file1: src, file2: dst}
    result_tuple = pivot_detector(file1, file2, df_dict)
    logging.info(f'PD Test Result: {result_tuple}')
    assert result_tuple[1]['pivot'] >= 1.0
