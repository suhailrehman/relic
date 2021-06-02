import glob

import networkx as nx

from relic.core import *
from relic.distance.nppo import join_detector, groupby_detector, pivot_detector
from relic.distance.tiebreakers import tiebreak_from_computed_scores, tiebreak_hash_edge

import pytest
import logging

from relic.graphs.clustering import get_graph_clusters_set

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/test_workflow/artifacts/')
expected_out_dir = os.path.join(THIS_DIR, 'data/test_workflow/inferred/')
artifact_set = set(os.path.basename(f) for f in glob.glob(data_dir+'/*.csv'))
file1 = '0.csv'
file2 = '1.csv'


@pytest.fixture(scope='session')
def relic_instance(tmpdir_factory):
    """Returns relic initialized to test_dir and a random output dir"""
    d = tmpdir_factory.mktemp('inferred_actual')
    logging.info('Testing RELIC on input:' + str(data_dir))
    logging.info('Temporary Output directory: '+str(d))
    return RelicAlgorithm(data_dir, d, name='test_workflow')


def test_create_initial_graph(relic_instance):
    logging.info('Testing initial graph creation')
    initial_graph = relic_instance.create_initial_graph()
    assert nx.is_empty(initial_graph)
    expected_nodes = artifact_set
    actual_nodes = set(e for e in initial_graph.nodes())
    assert len(actual_nodes) != 0
    assert expected_nodes == actual_nodes


def test_set_initial_clusters(relic_instance):
    logging.info('Testing initial cluster allocation')
    expected_clusters = set(frozenset(x) for x in relic_instance.set_initial_clusters().values())
    actual_clusters = get_graph_clusters_set(expected_out_dir+'clusters_with_filename.csv')
    assert len(expected_clusters) != 0
    assert expected_clusters == actual_clusters


def test_add_ppo_edges(relic_instance):
    logging.info('Testing PPO Cell Jaccard Edges')
    relic_instance.compute_edges_of_type(edge_type='all', similarity_function=compute_all_ppo_labels, n_pairs=2)
    relic_instance.add_edges_of_type(edge_type='jaccard', intra_cluster=True, tiebreak_function=tiebreak_from_computed_scores,
                                     final_function=tiebreak_hash_edge,
                                     tiebreak_kwargs={'pairwise_weights' : relic_instance.pairwise_weights,
                                                      'score_type': 'overlap'}
                                     )
    expected_graph = nx.read_edgelist(expected_out_dir+'infered_mst_pc2cell+containment+group+join+pivot.csv')
    expected_edge_set = set(frozenset((u, v)) for u, v, data in expected_graph.edges(data=True) if data['type'] == 'cell')
    actual_edge_set = set(frozenset((u,v)) for u, v, data in relic_instance.g_inferred.edges(data=True) if data['type'] == 'jaccard')
    assert expected_edge_set == actual_edge_set


def test_add_join_edges(relic_instance):
    logging.info('Testing Join Edges')
    relic_instance.compute_edges_of_type(edge_type='join', similarity_function=join_detector, n_pairs=3)
    relic_instance.add_edges_of_type(edge_type='join', intra_cluster=False, sim_threshold=1.0,
                                     tiebreak_function=tiebreak_from_computed_scores,
                                     final_function=tiebreak_hash_edge,
                                     tiebreak_kwargs={'pairwise_weights' : relic_instance.pairwise_weights,
                                                      'score_type': 'overlap'}
                                     )
    expected_graph = nx.read_edgelist(expected_out_dir+'infered_mst_pc2cell+containment+group+join+pivot.csv')
    expected_edge_set = set(frozenset((u, v)) for u, v, data in expected_graph.edges(data=True) if data['type'] == 'join')
    actual_edge_set = set(frozenset((u, v)) for u, v, data in relic_instance.g_inferred.edges(data=True) if data['type'] == 'join')
    logging.info([x for x in relic_instance.g_inferred.edges(data=True)])
    assert expected_edge_set == actual_edge_set


def test_inter_cell_edges(relic_instance):
    logging.info('Testing Intra_cell Edges')
    # relic_instance.compute_edges_of_type(edge_type='join', similarity_function=join_detector, n_pairs=3)
    relic_instance.add_edges_of_type(edge_type='jaccard', intra_cluster=False, sim_threshold=0.1,
                                     tiebreak_function=tiebreak_from_computed_scores,
                                     final_function=tiebreak_hash_edge,
                                     tiebreak_kwargs={'pairwise_weights' : relic_instance.pairwise_weights,
                                                      'score_type': 'overlap'}
                                     )
    expected_graph = nx.read_edgelist(expected_out_dir+'infered_mst_pc2cell+containment+group+join+pivot.csv')
    expected_edge_set = set(frozenset((u, v)) for u, v, data in expected_graph.edges(data=True) if 'cell' in data['type'])
    actual_edge_set = set(frozenset((u,v)) for u, v, data in relic_instance.g_inferred.edges(data=True) if data['type'] == 'jaccard')
    assert actual_edge_set == expected_edge_set


def test_groupby_edges(relic_instance):
    logging.info('Testing GroupBy Edges')
    relic_instance.compute_edges_of_type(edge_type='groupby', similarity_function=groupby_detector, n_pairs=2)
    relic_instance.add_edges_of_type(edge_type='groupby', intra_cluster=False, sim_threshold=1.0,
                                     tiebreak_function=tiebreak_from_computed_scores,
                                     final_function=tiebreak_hash_edge,
                                     tiebreak_kwargs={'pairwise_weights' : relic_instance.pairwise_weights,
                                                      'score_type': 'containment'}
                                     )
    expected_graph = nx.read_edgelist(expected_out_dir+'infered_mst_pc2cell+containment+group+join+pivot.csv')
    expected_edge_set = set(frozenset((u, v)) for u, v, data in expected_graph.edges(data=True) if data['type'] == 'groupby')
    actual_edge_set = set(frozenset((u,v)) for u, v, data in relic_instance.g_inferred.edges(data=True) if data['type'] == 'groupby')
    assert actual_edge_set == expected_edge_set


def test_pivot_edges(relic_instance):
    logging.info('Testing Pivot Edges')
    relic_instance.compute_edges_of_type(edge_type='pivot', similarity_function=pivot_detector, n_pairs=2)
    relic_instance.add_edges_of_type(edge_type='pivot', intra_cluster=False, sim_threshold=1.0,
                                     tiebreak_function=tiebreak_from_computed_scores,
                                     final_function=tiebreak_hash_edge,
                                     tiebreak_kwargs={'pairwise_weights' : relic_instance.pairwise_weights,
                                                      'score_type': 'containment'}
                                     )
    expected_graph = nx.read_edgelist(expected_out_dir+'infered_mst_pc2cell+containment+group+join+pivot.csv')
    expected_edge_set = set(frozenset((u, v)) for u, v, data in expected_graph.edges(data=True) if data['type'] == 'pivot')
    actual_edge_set = set(frozenset((u,v)) for u, v, data in relic_instance.g_inferred.edges(data=True) if data['type'] == 'pivot')
    assert actual_edge_set == expected_edge_set

