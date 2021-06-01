import os
from relic.algorithm import *
from relic.utils.serialize import build_df_dict_dir
from relic.distance.ppo import compute_all_ppo
import pytest
import logging

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/test_workflow/artifacts/')
file1 = '1.csv'
file2 = '2.csv'
expected_max = [(frozenset(('0.csv', '1.csv')), 0.9999523809523809)]


def test_compute_tuplewise_similarity():
    logging.info('Input and output directories:' + THIS_DIR + data_dir)
    dataset = build_df_dict_dir(data_dir)
    sim_dict = compute_tuplewise_similarity(dataset, similarity_metric=compute_all_ppo)
    test_pair = (file1, file2)
    logging.debug('PQDict Contents: ' + str(sim_dict['jaccard']))
    max_vals = sim_dict['jaccard'].pop_max()
    logging.info('PQDict Maxedges: ' + str(max_vals))
    assert sim_dict['jaccard'][test_pair] == pytest.approx(0.8, rel=1e-3)
    max_edges = set(frozenset(e[0]) for e in max_vals)
    expected_max_edges = set(frozenset(e[0]) for e in expected_max)
    assert max_edges == expected_max_edges
