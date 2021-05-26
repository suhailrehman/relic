import os
from relic.algorithm import *
from relic.utils.serialize import build_df_dict_dir
from relic.distance.ppo import compute_all_ppo
import pytest
import logging

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/20210126-153738/artifacts/')
file1 = '0.csv'
file2 = '1.csv'
expected_max = [(frozenset(('5.csv', '3.csv')), 0.9988888888888889), (frozenset(('6.csv', '3.csv')), 0.9988888888888889)]


def test_compute_tuplewise_similarity():
    logging.info('Input and output directories:' + THIS_DIR + data_dir)
    dataset = build_df_dict_dir(data_dir)
    sim_dict = compute_tuplewise_similarity(dataset, similarity_metric=compute_all_ppo)
    test_pair = frozenset((file1, file2))
    logging.debug('PQDict Contents: '+ str(sim_dict['jaccard']))
    max_vals = sim_dict['jaccard'].pop_max()
    logging.info('PQDict Maxedges: ' + str(max_vals))
    assert sim_dict['jaccard'][test_pair] == pytest.approx(0.967)
    max_edges = set(e[0] for e in max_vals)
    expected_max_edges = set(e[0] for e in expected_max)
    assert max_edges == expected_max_edges