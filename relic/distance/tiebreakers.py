from hashlib import md5
from relic.utils.pqedge import PQEdges
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def hash_edge(x):
    w = "+".join(sorted(list(x[0]))).encode('utf8')
    return md5(w).hexdigest()


def hash_edge_join(x):
    """ Example Edge: ((('1_022.csv', '1_014.csv'), '1_021.csv'), 1.0)"""
    w = "+".join((x[0][0][0], x[0][0][1], x[0][1])).encode('utf8')
    return md5(w).hexdigest()


def tiebreak_hash_edge(edges):
    return sorted(edges, key=hash_edge)[0]


def tiebreak_hash_edge_join(edges):
    return sorted(edges, key=hash_edge_join)[0]


def tiebreak_from_computed_scores(edge_list, pairwise_weights=None, score_type=None):
    pqe = PQEdges()
    for ed, score in edge_list:
        try:
            pqe.additem(ed, pairwise_weights[score_type][ed])
        except KeyError:
            logger.warning(f'Missing Score for Tie Break: {ed}')
    return pqe.pop_max()


def tiebreak_join_from_inferred_graph(edge_list, g_inferred=None):
    pqe = PQEdges()
    for ed, score in edge_list:
        src1, src2, dst = ed[0][0], ed[0][1], ed[1]
        current_joins = sum(1 for e in g_inferred.edges(dst, data=True) if e[2]['type'] == 'join')
        if current_joins < 2:
            pqe.additem(ed, score)
    return pqe.pop_max()