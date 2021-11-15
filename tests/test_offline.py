from itertools import combinations

from relic.offline import *


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
wf_dir = os.path.join(THIS_DIR, 'data/test_workflow/')
data_dir = os.path.join(wf_dir, '/artifacts/')


def test_enumeration(tmpdir):
    output = tmpdir.mkdir("inferred_actual").join("pairs.csv")
    enumerate_tuple_pairs(data_dir, output, 2)
    artifact_set = set(os.path.basename(f) for f in glob.glob(data_dir + '/*.csv'))
    expected_pair_set = set(frozenset((u, v)) for u, v in combinations(artifact_set, 2))

    with open(output, 'r') as fp:
        actual_pair_set = set([frozenset(x.strip().split(',')) for x in fp.readlines()])

    assert actual_pair_set == expected_pair_set


def test_join_enumeration(tmpdir):
    assert True


def test_compute_distance_pair():
    assert True


def test_combine_and_create_pkl():
    assert True


def test_enumerate_gt_op_tuples(tmpdir):
    output = tmpdir.mkdir("offline").join("join_scores.csv")
    gt_graph_file = wf_dir+'dataset_gt_fixed.pkl'
    enumerate_gt_op_tuples(gt_graph_file=gt_graph_file,
                           op_type='join',
                           filename=output)
    assert os.path.exists(output)
