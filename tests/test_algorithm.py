import os
from relic.algorithm import *
from relic.utils.serialize import build_df_dict_dir
from relic.distance.ppo import compute_all_ppo
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/20210126-153738/artifacts/')
file1 = '0.csv'
file2 = '1.csv'


def test_get_tuplewise_similarity():
    print(THIS_DIR, data_dir)
    dataset = build_df_dict_dir(data_dir)
    sim_dict = get_tuplewise_similarity(dataset, similarity_metric=compute_all_ppo)
    test_pair = frozenset((file1, file2))
    print(sim_dict['jaccard'])
    max_vals = sim_dict['jaccard'].pop_max()
    print(max_vals)
    assert sim_dict['jaccard'][test_pair] == pytest.approx(0.967)
