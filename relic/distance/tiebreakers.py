from hashlib import md5

from relic.distance.nppo import get_group_agg_cols
from relic.distance.ppo import compute_all_ppo
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


def tiebreak_hash_edge_pqe(edges, pqe=None):
    for ed in edges:
        if ed not in pqe:
            pqe.additem(ed, hash_edge(ed))
    return pqe


def tiebreak_hash_edge_join_pqe(edges, pqe=None):
    for ed in edges:
        if ed not in pqe:
            pqe.additem(ed, hash_edge_join(ed))
    return pqe


def tiebreak_from_computed_scores(edge_list, pqe=None, pairwise_weights=None, score_type=None):
    for ed, score in edge_list:
        try:
            if ed not in pqe:
                pqe.additem(ed, pairwise_weights[score_type][ed])
        except KeyError:
            logger.warning(f'Missing Score for Tie Break: {ed}')
    return pqe


def tiebreak_join_from_inferred_graph(edge_list, pqe=None, g_inferred=None):
    for ed, score in edge_list:
        src1, src2, dst = ed[0][0], ed[0][1], ed[1]
        current_joins = sum(1 for e in g_inferred.edges(dst, data=True) if e[2]['type'] == 'join')
        if current_joins < 2 and ed not in pqe:
            pqe.additem(ed, score)
    return pqe


def tiebreak_join_computed_scores(edge_list, pqe=None, pairwise_weights=None, score_type=None):
    for ed, score in edge_list:
        src1, src2, dst = ed[0][0], ed[0][1], ed[1]
        new_score = pairwise_weights[score_type][frozenset([dst, src1])] * pairwise_weights[score_type][frozenset([dst, src2])]
        if ed not in pqe:
            pqe.additem(ed, new_score)
    return pqe


def tiebreak_join_src_containment(edge_list, pqe=None, df_dict=None):
    for ed, score in edge_list:
        src1, src2, dst = ed[0][0], ed[0][1], ed[1]
        new_score = compute_all_ppo(df_dict[src1], df_dict[dst], ppo='containment_oneside')['containment_oneside']
        new_score *= compute_all_ppo(df_dict[src2], df_dict[dst], ppo='containment_oneside')['containment_oneside']
        if ed not in pqe:
            pqe.additem(ed, new_score)
    return pqe


def tiebreak_groupby_replay(edge_list, df_dict=None, pqe=None):
    for ed, score in edge_list:
        src1, src2 = ed
        if ed not in pqe.keys():
            logger.debug(f'Adding GB edge to PQE {ed}, pqe')
            score = replay_groupby(df_dict[src1], df_dict[src2])
            pqe.additem(ed, score)
        else:
            logger.debug(f'GB Edge already in PQE: {ed}, pqe')
    return pqe


def replay_groupby(df1, df2, grouplist=None, agg_ops=['min', 'max', 'sum', 'mean', 'count'], debug=False):
    if not grouplist:
        grouplist, agg_cols, missing_vals = get_group_agg_cols(df1, df2)

    if len(df1.index) < len(df2.index):
        src = df2
        dst = df1
    else:
        src = df1
        dst = df2

    similarities = []

    # Order the groupcols based on order in the destination.
    ordered_grouplist = []
    for col in dst.columns:
        if col in grouplist:
            ordered_grouplist.append(col)

    group_object = src.groupby(ordered_grouplist)
    for agg_op in agg_ops:
        try:
            agg_func = getattr(group_object, agg_op)
            group_result = agg_func().reset_index()
            similarities.append(compute_all_ppo(dst, group_result, ppo='jaccard')['jaccard'])
            logger.debug(f"agg_op:{agg_op}, dst_df: {dst.head()}, group_result: {group_result.head()}")
        except Exception as e:
            logger.warning(f'Warning GroupBy op failed: {agg_op}, {e}')
            similarities.append(0.0)

    # if debug:
    #     print('GroupBy performed on column: ', ordered_grouplist)
    #     print('Columns, rows df1: ', df1.columns, len(df1.index))
    #     print('Columns, rows df2: ', df2.columns, len(df2.index))
    #     print('Columns, rows group result: ', group_result.columns, len(group_result.index))
    #     print('Similarities: ', similarities)
    return max(similarities)
