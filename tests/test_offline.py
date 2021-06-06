from itertools import combinations

from relic.offline import *

# @pytest.mark.parametrize("earned,spent,expected", [
#     (30, 10, 20),
#     (20, 2, 18),
# ])

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/test_workflow/artifacts/')


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


def
