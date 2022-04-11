from tqdm.auto import tqdm
from itertools import combinations, product
from collections import defaultdict
from math import comb
import logging

from relic.utils.pqedge import PQEdges
from relic.distance.ppo import compute_all_ppo_labels
from relic.utils.matching import schema_match_df_combo, schema_match_df_triple

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def compute_tuplewise_similarity(dataset, similarity_metric=compute_all_ppo_labels, threshold=-1.0,
                                 silent=False, pairs=None, n_pairs=2,
                                 tuplewise_similarity=None, label=None, match_schema=False, **kwargs):
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
        label: distance label to be used in relic (default None for PPO)
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
        logger.debug('Evaluating tuple: ' + str(tup))
        # tuple_dfs = [dataset[x] for x in tup]
        if match_schema:
            if len(tup) == 3:
                ds, sm = schema_match_df_triple(tup, dataset)
            else:
                ds, sm = schema_match_df_combo(tup, dataset)
            logger.debug(f"Schema Matches: {sm}")
        else:
            ds = dataset

        edge, scores_dict = similarity_metric(*tup, ds, **kwargs)

        for ppo_type, score in scores_dict.items():
            logger.debug(f'Adding {ppo_type} score: {scores_dict[ppo_type]} to edge {edge}')
            tuplewise_similarity[ppo_type].additem(edge, scores_dict[ppo_type])


        # scores_dict returns: ((srcs), dest)), {score_dict}

        # Previous code
        # if label in ['join', 'groupby', 'pivot']:
        #     # Special case handling for NPPO (not in dict form currently)
        #     logger.debug(f'Adding {label} score: {scores_dict}')
        #     tuplewise_similarity[label].additem((scores_dict[0], scores_dict[1]), scores_dict[2])
        # else:
        #     for ppo_type, score in scores_dict.items():
        #         if score >= threshold:
        #             tuplewise_similarity[ppo_type].additem(tup, scores_dict[ppo_type])
        #         else:
        #             logger.debug('Dropping tuple: ' + tup + ' below threshold ' + score)

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
