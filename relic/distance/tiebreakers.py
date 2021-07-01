from hashlib import md5
from relic.utils.pqedge import PQEdges
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def hash_edge(x):
    w = "+".join(sorted(list(x[0]))).encode('utf8')
    return md5(w).hexdigest()


def hash_edge_join(x):
    w = "+".join((x[0][0], x[0][1], x[1])).encode('utf8')
    return md5(w).hexdigest()


def tiebreak_hash_edge(edges):
    return sorted(edges, key=hash_edge)[0]


def tiebreak_hash_edge_join(edges):
    return sorted(edges, key=hash_edge_join)[0]


def tiebreak_from_computed_scores(edge_list, pairwise_weights=None, score_type=None):
    pqe = PQEdges()
    for ed, score in edge_list:
        pqe.additem(ed, pairwise_weights[score_type][ed])
    return pqe.pop_max()