from tqdm.auto import tqdm
from itertools import combinations, product
from collections import defaultdict
from math import comb
import logging

from relic.utils.pqedge import PQEdges
from relic.distance.ppo import compute_all_ppo

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def compute_tuplewise_similarity(dataset, similarity_metric=compute_all_ppo, threshold=-1.0,
                                 silent=False, pairs=None, n_pairs=2,
                                 tuplewise_similarity=None, **kwargs):
    """Compute pairwise similarity metrics of dataset dict using similarity_metric
    returns reverse sorted pqdict of frozenset(pair1, pair2) -> similarity_score mappings

    Args:
        dataset: dict containing artifact_name string : DataFrame mappings
        similarity_metric: function used to compute distance between dataframes df1 and df2
        threshold: minimum threshold required to be included in return. default -1.0 (all distances)
        silent: disable tqdm output (progress bar) default False
        pairs: custom iterable of pairs to be computed, default None (all pairs)
        n_pairs: number of tuples to evaluate (default 2)
        tuplewise_similarity: previous pqdict mapping to update
        kwargs: other keyword args to be passed to similarity_metric function.

    Returns: pqdict mapping {frozenset(pair1, pair2) -> similarity_score}
    """
    if not tuplewise_similarity:
        tuplewise_similarity = defaultdict(PQEdges)
    if not pairs:
        pairs = combinations(dataset.keys(), n_pairs)
        total_len = comb(len(dataset.keys()), n_pairs)
    else:
        total_len = len(pairs)
    for tup in tqdm(pairs, desc='graph pairs', leave=False, disable=silent, total=total_len):
        logger.debug('Evaluating tuple: ', tup)
        tuple_dfs = [dataset[x] for x in tup]
        scores_dict = similarity_metric(*tuple_dfs, **kwargs)
        for ppo_type, score in scores_dict.items():
            if score >= threshold:
                # TODO : add threshold dict of default thresholds for various PPOs
                tuplewise_similarity[ppo_type].additem(frozenset(tup), scores_dict[ppo_type])
            else:
                logger.debug('Dropping tuple: ', tup, ' below threshold ', score)

    return tuplewise_similarity


def _generate_tuples(sets):
    for tup in product(*sets):
        if len(set(tup)) != len(tup):
            continue
        yield tup


'''
def intra_cluster_similarity(df_dict, clusters, threshold=0.25):
    pairwise_jaccard = []
    for cluster in clusters.values():
        batch = {k: df_dict[k] for k in cluster}
        pw_batch = get_tuplewise_similarity(batch, compute_all_ppo, threshold=threshold)
        pairwise_jaccard.extend(pw_batch)
    return pairwise_jaccard
'''
