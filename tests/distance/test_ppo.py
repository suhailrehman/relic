import os
import pandas as pd
from relic.distance.ppo import *
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, '../data/20210126-153738/artifacts/')
file1 = '0.csv'
file2 = '1.csv'


def test_compute_all_ppo():
    df1 = pd.read_csv(data_dir+file1, index_col=0)
    df2 = pd.read_csv(data_dir+file2, index_col=0)
    result_dict = compute_all_ppo(df1, df2)
    assert result_dict['jaccard'] == pytest.approx(0.967)
    assert result_dict['containment'] == pytest.approx(0.967)
    assert result_dict['overlap'] == pytest.approx(967)
    assert result_dict['containment_oneside'] == pytest.approx(0.967)
