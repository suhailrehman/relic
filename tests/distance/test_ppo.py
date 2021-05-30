import os
import pandas as pd
from relic.distance.ppo import *
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, '../data/test_workflow/artifacts/')
file1 = '0.csv'
file2 = '1.csv'


def test_compute_all_ppo():
    df1 = pd.read_csv(data_dir+file1, index_col=0)
    df2 = pd.read_csv(data_dir+file2, index_col=0)
    df_dict = {file1: df1, file2: df2}
    result_dict = compute_all_ppo(file1, file2, df_dict)
    assert result_dict['jaccard'] == pytest.approx(0.999, rel=1e-3)
    assert result_dict['containment'] == pytest.approx(0.999, rel=1e-3)
    assert result_dict['overlap'] == pytest.approx(20999.0, rel=1e-3)
    assert result_dict['containment_oneside'] == pytest.approx(0.999, rel=1e-3)
