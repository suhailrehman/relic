#import pytest
from itertools import combinations

from pqedge import PQEdges

sample_artifacts = [
    '1.csv',
    '2.csv',
    '3.csv',
    '4.csv'
]

sample_scores = [1.0,1.0,0.998,0.9987,1.0,0.0]
edge_pairs = [frozenset([u,v]) for u,v in combinations(sample_artifacts, 2)]
edge_scores = {x: y for x, y in zip(edge_pairs, sample_scores)}

def test_pop_max():
    pq = PQEdges(data=edge_scores)
    ones = pq.pop_max()
    assert len(ones) == 3
    assert all([v[1] == 1.0 for v in ones])
    assert len(pq) == 3
    seconds = pq.pop_max()
    assert len(seconds) == 1
    assert seconds[0][1] == 0.9987

