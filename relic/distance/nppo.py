#!/usr/bin/env python

"""nppo.py: Detectors for Non-Point Preserving Operations."""

__author__ = "Suhail Rehman"
__email__ = "suhail@uchicago.edu"

import numpy as np
import itertools
import glob
import os

import networkx as nx

from collections import defaultdict

from relic.distance import ppo, set_functions
from relic.distance.tiebreakers import hash_edge_join, hash_edge

from relic.graphs import clustering
import csv

import pprint
import operator

from tqdm.auto import tqdm
from hashlib import md5

import math
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)





# Join Detection
from relic.utils.matching import get_common_cols, generate_common_lattice


def join_detector(df1, df2, df3, df_dict, debug=False, replay=True):
    if debug:
        print("Join Scoring:", df1, df2, df3)
    combo_set = set((df1, df2, df3))
    max_combo = None
    max_col_number = 0

    columns_dict = {df: set(df_dict[df].columns) for df in combo_set}

    # First Check for at least one common column in triple to act as key
    common_cols = columns_dict[df1].intersection(columns_dict[df2]).intersection(columns_dict[df3])

    if not common_cols:
        logging.debug(f'{df1}, {df2}, {df3}, No common cols')
        return ((df1, df2), df3, 0.0)

    for join_dest in combo_set:
        join_sources = combo_set - set([join_dest])

        # Whats the jaccard distance of the dest as a union of the sources?

        set_iterator = iter(join_sources)
        source = next(set_iterator)
        other_source = next(set_iterator)

        symm_diff = columns_dict[source].intersection(columns_dict[join_dest]) - columns_dict[other_source]
        column_union = symm_diff.union(
            columns_dict[other_source].intersection(columns_dict[join_dest]) - columns_dict[source])

        jaccard = set_functions.set_jaccard_similarity(columns_dict[join_dest], column_union) * len(
            columns_dict[join_dest])

        logging.debug(f'JD Debug: {source}, {other_source}, {join_dest}, {symm_diff}, {column_union}, {jaccard}')
        if jaccard > max_col_number:
            max_col_number = jaccard
            max_combo = (tuple(join_sources), join_dest)

    if not max_combo or not common_cols:
        logging.debug(f'{df1}, {df2}, {df3}, No Max Combo')
        return ((df1, df2), df3, 0.0)

    df1, df2 = max_combo[0]
    df3 = max_combo[1]

    df1_columns = columns_dict[df1].intersection(columns_dict[df3]) - columns_dict[df2]
    df2_columns = columns_dict[df2].intersection(columns_dict[df3]) - columns_dict[df1]

    # Check for non-key contributions from either side:
    if not df1_columns or not df2_columns:
        logging.debug(f'{df1}, {df2}, {df3}, Poor contribution from either side')
        return ((df1, df2), df3, 0.0)

    keys = columns_dict[df1].intersection(columns_dict[df2])

    containments = {}

    for join_key in keys:  # Assuming single column inner join
        # intersecting_values = set(frozenset(v) for v in df_dict[df1][join_key].values).intersection(
        #    set(frozenset(v) for v in df_dict[df2][join_key].values))

        # key_list = list(v for fset in intersecting_values for v in fset)

        key_list = set(df_dict[df1][join_key].values).intersection(set(df_dict[df2][join_key].values))

        df1_merge_vals = df_dict[df1].loc[df_dict[df1][join_key].isin(key_list)]
        df2_merge_vals = df_dict[df2].loc[df_dict[df2][join_key].isin(key_list)]

        df1_containment = col_group_containment(df1_merge_vals, df_dict[df3], df1_columns)
        df2_containment = col_group_containment(df2_merge_vals, df_dict[df3], df2_columns)

        containments[join_key] = df1_containment * df2_containment

        logging.debug(f'{df1}, {df2}, {df3}, Join Key: {join_key}, Containment: {containments[join_key]}')
            # print('Key List:', key_list)
            # print('df1_merge_vals', df1_merge_vals)
            # print('df2_merge_vales', df2_merge_vals)

    # print(df1_containment, df2_containment)

    max_containment = max(containments.items(), key=operator.itemgetter(1))[0]

    if containments[max_containment] < 0.001:
        logging.debug(f'{df1}, {df2}, {df3}, Poor max containment')
        return ((df1, df2), df3, 0.0)

    if debug:
        print('df1,df2,df3:', df1, df2, df3)
        print('column_sets:', df1_columns, df2_columns, columns_dict[df3])
        print('join key:', max_containment)

    df1_containment = col_group_containment(df_dict[df1], df_dict[df3], df1_columns)
    df2_containment = col_group_containment(df_dict[df2], df_dict[df3], df2_columns)

    if df1_containment < 0.01 or df2_containment < 0.01:
        logging.debug(f'{df1}, {df2}, {df3}, Poor single-sided containments')
        return ((df1, df2), df3, 0.0)

    # TODO: Check coherency here
    # Do Join replay for now:
    if replay:
        logging.debug(f'Replaying Join: {df1} (join) {df2} -> {df3} using key {max_containment}')
        try:
            replay_score1 = replay_merge(df_dict[df1], df_dict[df2], df_dict[df3], max_containment)
            replay_score2 = replay_merge(df_dict[df2], df_dict[df1], df_dict[df3], max_containment)
            replay_score = max(replay_score1, replay_score2)
        except (ValueError, pd.errors.MergeError) as e:
            logging.debug('Could not merge ', df1, df2, 'to produce', df3, 'using', max_containment)
            replay_score = 0.0
            pass
    else:
        replay_score = df1_containment * df2_containment


    logging.debug(f'max_col_ratio, maxcontain, replay_score: {max_col_number}, {containments[max_containment]}, {replay_score}')

    return (max_combo[0], max_combo[1], replay_score)


def get_max_coherent_columns_1(df1, df2):
    common_cols = get_common_cols(df1, df2)

    coherent_1_cols = set(col for col in common_cols if check_col_containment(df1, df2, col))

    if coherent_1_cols:
        if check_col_group_containment(df1, df2, coherent_1_cols):
            return coherent_1_cols

    return None


# Looks for perfect column containment of colname between dataframes df1, df2
def check_col_containment(df1, df2, colname, col2name=None):
    if (col2name == None):
        col2name = colname
    return set(df1[colname]).issubset(set(df2[col2name]))


def col_containment(df1, df2, colname, col2name=None):
    if (col2name == None):
        col2name = colname

    df1valset = set(df1[colname])
    df2valset = set(df2[col2name])

    if len(df2valset) == 0:
        return 0.0

    return len(df1valset.intersection(df2valset)) / len(df2valset)


# Looks for perfect colgroup containment of colgroup between dataframes df1, df2
def check_col_group_containment(df1, df2, colgroup, colgroup2=None):
    if (colgroup2 == None):
        colgroup2 = colgroup

    df1valset = set(frozenset(u) for u in df1[list(colgroup)].values.tolist())
    df2valset = set(frozenset(u) for u in df2[list(colgroup2)].values.tolist())

    # print(df2valset)

    return df1valset.issubset(df2valset)


def col_group_containment(df1, df2, colgroup, colgroup2=None, denom='df2'):
    # Rewrite: See if df2 is contained in df1
    if (colgroup2 == None):
        colgroup2 = colgroup

    # print(df1.columns)
    # print(df2.columns)
    df1valset = set(frozenset(u) for u in df1[list(colgroup)].values.tolist())
    df2valset = set(frozenset(u) for u in df2[list(colgroup2)].values.tolist())

    # print(df2valset)
    if len(df1valset.union(df2valset)) < 1:
        return 0.0

    if denom == 'df2':
        return len(df1valset.intersection(df2valset)) / len(df2valset)
    else:
        return len(df1valset.intersection(df2valset)) / len(df1valset)


def explore_group_lattice(df1, df2, group_cols, jaccard_threshold=1.0, maxdepth=3):
    def col_group_jaccard(d1, d2, colgroup):
        df1valset = set(frozenset(u) for u in d1[list(colgroup)].values.tolist())
        df2valset = set(frozenset(u) for u in d2[list(colgroup)].values.tolist())

        if len(df1valset.union(df2valset)) < 1:
            return 0.0

        return len(df1valset.intersection(df2valset)) / len(df1valset.union(df2valset))

    def restofsubsets(goodsubset, remainingels, condition, maxdepth):
        answers = []
        for j in range(len(remainingels)):
            nextsubset = goodsubset + remainingels[j:j + 1]
            if len(nextsubset) <= maxdepth and condition(nextsubset):
                answers.append(nextsubset)
                answers += restofsubsets(nextsubset, remainingels[j + 1:], condition)
        return answers

    lattice = restofsubsets([], group_cols, lambda l: col_group_jaccard(df1, df2, l) >= jaccard_threshold, maxdepth)

    return max(lattice, key=lambda x: len(x))


## Group By Detection

def get_group_agg_cols(df1, df2, contaiment_threshold=0.9, sim_threshold=0.9, lattice_check=False, debug=False):
    group_cols = []
    group_col_containment = []
    agg_cols = []

    # TODO: Use a column matching map instead
    common_cols = set(list(df1)).intersection(set(list(df2)))

    for col in common_cols:
        containment = col_containment(df1, df2, col)
        jsim = ppo.set_jaccard_similarity(set(df1[col].values), set(df2[col].values))
        logging.debug(f'Col, containment, jsim: {col}, {containment}, {jsim}')
        if containment >= contaiment_threshold and jsim >= sim_threshold:
            group_cols.append(col)
            group_col_containment.append(containment)

        else:
            agg_cols.append(col)

    logging.debug(f'Determining Group Cols:: {group_cols}, {group_col_containment}')

    if lattice_check and group_cols:
        group_cols = explore_group_lattice(df1, df2, group_cols, jaccard_threshold=sim_threshold)

    # Find the group value containment for the contained columns:
    srcvalset = set(frozenset(u) for u in df1[group_cols].values.tolist())
    dstvalset = set(frozenset(u) for u in df2[group_cols].values.tolist())

    logging.debug(f'Source Val set: {srcvalset}')
    logging.debug(f'Dest Val set: {dstvalset}')
    logging.debug(f'Symmdiff: {srcvalset.symmetric_difference(dstvalset)}')

    missing_vals = (len(srcvalset.symmetric_difference(dstvalset)) / len(srcvalset))

    return group_cols, agg_cols, missing_vals


def groupby_detector(d1, d2, df_dict, debug=False, strict_schema=False, lattice_check=False,
                     null_aggs=False):
    # TODO: Column matching
    df1 = df_dict[d1]
    df2 = df_dict[d2]
    common_cols = set(list(df1)).intersection(set(list(df2)))

    # Strict Schema Check
    sym_diff_cols = set(list(df1)).symmetric_difference(set(list(df2)))
    if sym_diff_cols and strict_schema:
        logging.debug(f'GB({d1},{d2}): Failed Strict schema check')
        return d1, d2, 0.0

    if not common_cols:
        logging.debug(f'GB({d1},{d2}): DFs have no common columns')
        return d1, d2, 0.0

    # Contraction Check:
    if len(df1.index) == len(df2.index):
        logging.debug(f'GB({d1},{d2}): There is no len contraction')
        return d1, d2, 0.0

    src, dst = ((df1, df2) if len(df1.index) > len(df2.index) else (df2, df1))

    # Containment check and dividing columns into group and aggregate
    group_cols, agg_cols, missing_vals = get_group_agg_cols(src, dst, sim_threshold=1.0,
                                                            lattice_check=lattice_check, debug=debug)

    logging.debug(f'GB({d1},{d2}): Detected Group Cols: {group_cols}')
    logging.debug(f'GB({d1},{d2}): Detected Agg Cols: {agg_cols}')
    logging.debug(f'GB({d1},{d2}): Missing Values: {missing_vals}')

    if not group_cols:
        logging.debug(f'GB({d1},{d2}): No group columns detected')
        return d1, d2, 0.0

    if not agg_cols and not null_aggs:  # Don't allow aggregates to be null
        logging.debug(f'GB({d1},{d2}): No agg cols in common between the dataframes)')
        return d1, d2, 0.0

    if missing_vals > 0.0:
        logging.debug(f'GB({d1},{d2}): GroupBy has missing values')
        return d1, d2, 0.0

    # TODO: Lattice exploration
    src_group_keyness_ratio = columnset_keyness_ratio(src, group_cols)
    dst_group_keyness_ratio = columnset_keyness_ratio(dst, group_cols)

    if src_group_keyness_ratio == 1.0:
        logging.debug(f'GB({d1},{d2}): Source group columns are also keys, groupby unlikely: {src_group_keyness_ratio}')
        return d1, d2, 0.0

    if dst_group_keyness_ratio < 1.0:
        logging.debug(f'GB({d1},{d2}): Group keyness below threshold: {dst_group_keyness_ratio}')
        return d1, d2, 0.0

    column_diff = set_functions.set_jaccard_similarity(set(df1.columns), set(df2.columns))
    contraction_ratio = len(src.index) / len(dst.index)
    final_val = ((1.0 * len(group_cols) * dst_group_keyness_ratio) - missing_vals)  # * column_diff

    logging.debug(f'GB({d1},{d2}): final_val = (1.0 * len(group_cols) * group_keyness_ratio) - missing_vals')
    logging.debug(f'GB({d1},{d2}): {final_val} = (1.0 * {len(group_cols)} * {dst_group_keyness_ratio} - {missing_vals}')

    return d1, d2, final_val  # * contraction_ratio


#### Groupby


def columnset_keyness_ratio(df, colset):
    original_size = len(df[colset].index)
    set_size = len(set(tuple(u) for u in df[colset].values.tolist()))

    return set_size / original_size


## Transform Detector

def transform_detector(df1_name, df2_name, df_dict, g_inferred):
    df1 = df_dict[df1_name]
    df2 = df_dict[df2_name]

    index_name_match = True
    if df1.index.name and df2.index.name:
        if df1.index.name != df2.index.name:
            index_name_match = False

    if not index_name_match:
        return 0.0

    index_ratio = ppo.set_jaccard_similarity(set(df1.index), set(df2.index))
    column_ratio = ppo.set_max_containment(set(df1.columns), set(df2.columns))
    cell_score = 1.0 - ppo.compute_jaccard_DF(df1, df2)

    # print(index_ratio, column_ratio, cell_score)

    return index_ratio * column_ratio * cell_score


### Common NPPO component search routine:


def find_components_nppo_edge(g_inferred, df_dict, edge_num, nppo_dict, replay_dict=None,
                              nppo_function=groupby_detector,
                              label='groupby',
                              threshold=1.0, g_truth=None):
    components = [c for c in nx.connected_components(g_inferred)]
    print('components: ', len(components))

    all_cmp_pairs_similarties = []

    for srccmp, dstcmp in tqdm(itertools.combinations(components, 2), total=math.comb(len(components), 2)):
        # Group Edges Checked here
        similarites, nppo_dict = get_pairs_nppo_edges(df_dict, srccmp, dstcmp, g_inferred, nppo_function, nppo_dict)
        all_cmp_pairs_similarties.extend(similarites)

    if not all_cmp_pairs_similarties:
        return None, edge_num, nppo_dict, None, all_cmp_pairs_similarties, replay_dict

    # TODO: Common scoring function. right now 1.0 for all edges found.
    all_cmp_pairs_similarties.sort(key=lambda x: x[2], reverse=True)

    # print('NNPOs detected')
    # print(all_cmp_pairs_similarties)

    score_dict = clustering.generate_score_dict(all_cmp_pairs_similarties)
    maxscore = max(score_dict)

    if maxscore < threshold:
        print('No more edges above threshold')
        return None, edge_num, nppo_dict, None, all_cmp_pairs_similarties, replay_dict

    if len(score_dict[maxscore]) > 1:
        '''
        # Removing ground truth capabilities here.
        if nppo_function == df_groupby_check_new:
            print("Breaking tie by groupby replay", score_dict[maxscore])
            src, dst, score = tiebreak_by_groupby_replay(df_dict, score_dict[maxscore], g_truth=g_truth)

        elif nppo_function == pivot_detector: # Consider pivot tiebreaker to be GT group edge if present.
            print("Adding GT edge if present for pivot")
            found = False
            for u, v, s in score_dict[maxscore]:
                if g_truth.to_undirected().has_edge(u,v):
                    src, dst, score = u,v,s
                    found=True
            if not found:
                return None, edge_num, nppo_dict, None

        else:
        '''

        if nppo_function == groupby_detector:
            # print("Breaking tie by column-level", score_dict[maxscore])
            # src, dst, score = tiebreak_by_col_level(df_dict, score_dict[maxscore])
            print("Breaking tie by groupby-replay", score_dict[maxscore])
            src, dst, score, replay_dict = tiebreak_by_groupby_replay(df_dict, score_dict[maxscore], replay_dict)
        else:
            print("Breaking tie by contraction ratio", score_dict[maxscore])
            src, dst, score = tie_break_by_contraction_ratio(df_dict, score_dict[maxscore])

    else:
        src, dst, score = score_dict[maxscore][0]

    print('Adding nppo/group edge', src, dst, score)
    g_inferred.add_edge(src, dst, weight=score, num=edge_num, type=label)
    edge_num += 1
    return g_inferred, edge_num, nppo_dict, (src, dst), all_cmp_pairs_similarties, replay_dict


def augment_triples(triples, g_inferred):
    new_edges = []

    for d1, d2, d3 in triples:
        for n in g_inferred.neighbors(d1):
            new_edges.append(frozenset((n, d1, d2)))
            new_edges.append(frozenset((n, d1, d3)))
        for n in g_inferred.neighbors(d2):
            new_edges.append(frozenset((n, d2, d1)))
            new_edges.append(frozenset((n, d2, d3)))
        for n in g_inferred.neighbors(d3):
            new_edges.append(frozenset((n, d3, d1)))
            new_edges.append(frozenset((n, d3, d2)))

    print('Raw triples: ', len(new_edges))
    union = triples.union(set(new_edges))
    print('After union: ', len(union))
    return union


def augment_tuples(tuples, g_inferred):
    new_edges = set()

    for d1, d2 in tuples:
        for n in g_inferred.neighbors(d1):
            new_edges.add(frozenset((n, d1, d2)))
        for n in g_inferred.neighbors(d2):
            new_edges.add(frozenset((n, d2, d1)))

    # print('Raw triples: ', len(new_edges))
    # union = tuples.union(set(new_edges))
    # print('After union: ', len(union))
    return new_edges


def select_max_join_score(score_dict, g_inferred, threshold):
    src, dst, score = None, None, None

    for score in sorted(score_dict.keys(), reverse=True):
        if score >= threshold:
            if len(score_dict[score]) > 1:
                print("Warning multiple join-edges with same score", score_dict[score])
            s_edge_list = sorted(score_dict[score], key=hash_edge_join)
            for src_c, dst_c, score_c in s_edge_list:
                current_joins = sum(1 for e in g_inferred.edges(dst_c, data=True) if e[2]['type'] == 'join')
                if current_joins < 2:
                    return src_c, dst_c, score_c
                else:
                    print('Skipping ', src_c, dst_c, score_c, 'as destination is already part of join')
        else:
            break

    return src, dst, score



def find_components_join_edge(g_inferred, df_dict, edge_num, triple_dict, threshold=0.999):
    components = [c for c in nx.connected_components(g_inferred)]
    if len(components) > 2:
        combos = [[frozenset(i) for i in itertools.product(*c)] for c in itertools.combinations(components, 3)]
        triples = set(item for sublist in combos for item in sublist)
        print(len(triples), "number of join combinations to be explored")
        # print(triples)
        triples = augment_triples(triples, g_inferred)
        print(len(triples), "number of join combinations to be explored after augmentation")
    else:
        # Only two components:
        combos = [[frozenset(i) for i in itertools.product(*c)] for c in itertools.combinations(components, 2)]
        tuples = set(item for sublist in combos for item in sublist)
        print(len(tuples), "number of join combinations to be explored")
        # print(triples)
        triples = augment_tuples(tuples, g_inferred)
        print(len(triples), "number of join combinations to be explored after augmentation")

        # print(triples)

    join_scores = []

    for d1, d2, d3 in tqdm(triples):
        # print(tuple(fset))
        # d1,d2,d3 = tuple(fset)
        if frozenset((d1, d2, d3)) in triple_dict:
            similarities = triple_dict[frozenset((d1, d2, d3))]
        else:
            similarities = join_detector(d1, d2, d3, df_dict)
            triple_dict[frozenset((d1, d2, d3))] = similarities

        join_scores.append(similarities)

    if not join_scores:
        return None, edge_num, triple_dict, None, None

    # print('NNPOs detected')
    # print(join_scores)

    score_dict = clustering.generate_score_dict(join_scores)
    maxscore = max(score_dict)

    if maxscore <= threshold:
        return None, edge_num, triple_dict, None, None

    src, dst, score = select_max_join_score(score_dict, g_inferred, threshold)

    if None in (src, dst, score):
        return None, edge_num, triple_dict, None, None

    if g_inferred.has_edge(src[0], dst):
        print('Join Edge already present: ', src[0], dst)

    print('Adding nppo/group edge', src[0], dst, score)
    g_inferred.add_edge(src[0], dst, weight=score, num=edge_num, type='join')
    edge_num += 1

    if g_inferred.has_edge(src[1], dst):
        print('Join Edge already present: ', src[1], dst)

    print('Adding nppo/group edge', src[1], dst, score)
    g_inferred.add_edge(src[1], dst, weight=score, num=edge_num, type='join')

    edge_num += 1

    return g_inferred, edge_num, triple_dict, (src[0], dst), (src[1], dst)


def get_pairs_nppo_edges(dataset, cluster_set1, cluster_set2, g_inferred, nppo_function, nppo_dict, threshold=0.6):
    pairwise_similarity = []
    pairs = list(itertools.product(cluster_set1, cluster_set2))
    for d1, d2 in pairs:
        if d1 == d2:
            continue
        if frozenset((d1, d2)) in nppo_dict:
            score = nppo_dict[frozenset((d1, d2))]
        else:
            score = nppo_function(d1, d2, dataset, g_inferred)
            nppo_dict[frozenset((d1, d2))] = score
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            pass
            # print("WARNING: DROPPING",d1,d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity, nppo_dict


def compute_contraction_ratio(df_dict, df1_name, df2_name):
    df1 = df_dict[df1_name]
    df2 = df_dict[df2_name]

    if len(df1.index) > len(df2.index):
        srcdf = df1
        dstdf = df2
    else:
        srcdf = df2
        dstdf = df1

    return len(srcdf.index) / len(dstdf.index)


def tie_break_by_contraction_ratio(df_dict, pairlist):
    max_contraction = None
    max_score = 0

    scores_list = []

    for src, dst, score in pairlist:
        contraction = compute_contraction_ratio(df_dict, src, dst)
        scores_list.append((src, dst, score, contraction))

        if contraction >= max_score:
            max_contraction = (src, dst, score)
            max_score = contraction

    score_dict = clustering.generate_tiebreak_score_dict(scores_list)

    if len(score_dict[max_score]) > 1:
        print("Multiple Contraction candidates:", score_dict[max_score])
        s_edge_list = sorted(score_dict[max_score], key=hash_edge)
        return s_edge_list[0]

    return max_contraction


def find_column_set_match(df, valueset, string_convert=False):
    col_mapping_dict = defaultdict(list)

    for col in df.columns:
        if string_convert:

            col_data = set(df[col].astype('str').values)
            valueset = set(str(x) for x in valueset)
        else:
            col_data = set(df[col].values)
        score = ppo.set_jaccard_similarity(valueset, col_data)
        col_mapping_dict[score].append(col)

    return col_mapping_dict


def pivot_detector(df1_name, df2_name, df_dict, g_inferred=None, match_values=True, schema_check=True, debug=False,
                   index_match_threshold=0.99, col_match_threshold=0.99, replay_threshold=0.99):
    df1 = df_dict[df1_name]
    df2 = df_dict[df2_name]

    index_name_match = None
    src = None
    dst = None
    index_containment = 0.0
    max_col_score = 0.0
    max_col_mapping = None

    logging.debug(f'PD: df1index, df2index, df1columns, df2columns: {df1.index.name}, {df2.index.name}, {df1.columns}, {df2.columns}')
    if schema_check:  # Check for at most one common column (index)
        common_cols = set(df1).intersection(set(df2))
        if len(common_cols) > 1:
            return df1_name, df2_name, 0.0

    if df2.index.name:
        if df2.index.name in df1.columns:
            index_name_match = df2.index.name
            src = df1
            dst = df2

    elif df1.index.name:
        if df1.index.name in df2.columns:
            index_name_match = df1.index.name
            src = df2
            dst = df1

    if index_name_match:
        index_containment = ppo.set_jaccard_similarity(set(src[index_name_match].values), set(dst.index.values))
        logging.debug(f'Index name match & containment: {index_name_match}, {index_containment}')

    # otherwise use containment to figure out src and dst:
    else:
        return df1_name, df2_name, 0.0
        '''
        try:
            df1coldict = find_column_set_match(df1, set(df2.index.values))
            df2coldict = find_column_set_match(df2, set(df1.index.values))

            # print(df1coldict)
            # print(df2coldict)

            maxdf1 = max(df1coldict.keys())
            maxdf2 = max(df2coldict.keys())

            if maxdf1 > maxdf2:
                src = df1
                dst = df2
                index_containment = maxdf1
            else:
                src = df2
                dst = df1
                index_containment = maxdf2

        except ValueError as e:
            return 0.0
        '''

    if index_containment <= index_match_threshold:
        return df1_name, df2_name, 0.0

    # Check column-value containment
    try:
        coldict = find_column_set_match(src, set(dst), string_convert=True)
        max_col_score = max(coldict.keys())
        logging.debug(f'Column match dict: {coldict}')
    except ValueError as e:
        return df1_name, df2_name, 0.0

    # print(index_containment, max_col_score)
    if max_col_score <= col_match_threshold:
        return df1_name, df2_name, 0.0

    # Check pivot value containment if flag is set:
    if match_values:
        flat_values = dst.values.flatten()

        if flat_values.dtype == 'float64':
            valdict = find_column_set_match(src, set([x for x in flat_values if ~np.isnan(x)]))
        else:
            valdict = find_column_set_match(src, set([x for x in flat_values]))

        max_val_score = max(valdict.keys())
        logging.debug(f'Value column match: {valdict}')
        logging.debug(f'Pivot scores: {index_containment}, {max_col_score}, {max_val_score}')

        # Pivot replay section
        # index: src->index_name_match
        # col: coldict['max_col_score']
        # val: valdict['max_val_Score']
        if (index_containment * max_col_score) + max_val_score > replay_threshold:
            similarities = []
            for col in coldict[max_col_score]:
                for val in valdict[max_val_score]:
                    try:
                        replaydf = src.pivot_table(index=index_name_match, columns=col, values=val, aggfunc=max)
                        similarities.append(ppo.compute_all_ppo(dst, replaydf, None)['jaccard'])
                    except Exception as e:
                        logging.warning(f'Cannot pivot:', df1_name, df2_name, index_name_match, col, val, e)
                        similarities.append(0.0)
                logging.debug(f'pivot replay result: {dst.head()}, {replaydf.head()}, {similarities}')
            return df1_name, df2_name, max(similarities)

    return df1_name, df2_name, index_containment * max_col_score


def tiebreak_by_timestamp_synthetic(df_dict, pairlist):
    min_ts_diff = None
    min_diff = np.inf

    scores_list = []

    for src, dst, score in pairlist:
        src_time = int(src.split('.csv')[0])
        dst_time = int(dst.split('.csv')[0])

        time_diff = abs(dst_time - src_time)

        scores_list.append((src, dst, time_diff))

        if time_diff <= min_diff:
            min_ts_diff = (src, dst, score)
            min_diff = time_diff

    score_dict = clustering.generate_score_dict(scores_list)

    if len(score_dict[min_diff]) > 1:
        print("Multiple Timestamp candidates:", score_dict[min_diff])
        s_edge_list = sorted(score_dict[min_diff], key=hash_edge)
        return s_edge_list[0]

    return min_ts_diff


def tiebreak_by_groupby_replay(df_dict, pairlist, replay_dict):
    max_replay_candidate = None
    max_cell_score = 0.0

    scores_list = []

    for src, dst, score in pairlist:
        if frozenset((src, dst)) in replay_dict:
            replay_score = replay_dict[frozenset((src, dst))]
        else:
            replay_score = replay_groupby(df_dict[src], df_dict[dst])
            replay_dict[frozenset((src, dst))] = replay_score

        scores_list.append((src, dst, replay_score))

        if replay_score >= max_cell_score:
            max_replay_candidate = (src, dst, replay_score)
            max_cell_score = replay_score

    score_dict = clustering.generate_score_dict(scores_list)

    if len(score_dict[max_cell_score]) > 1:
        print("Multiple Replay candidates:", score_dict[max_cell_score])
        s_edge_list = sorted(score_dict[max_cell_score], key=hash_edge)
        return *s_edge_list[0], replay_dict

    return *max_replay_candidate, replay_dict


def replay_groupby(df1, df2, grouplist=None,
                   agg_ops=['min', 'max', 'sum', 'mean', 'count'], debug=False):
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
            similarities.append(ppo.compute_jaccard_DF(dst, group_result))
            if debug:
                print(agg_op)
                print(dst.head())
                print(group_result.head())
        except Exception as e:
            print('Warning GroupBy op failed: ', agg_op, e)
            similarities.append(0.0)

    if debug:
        print('GroupBy performed on column: ', ordered_grouplist)
        print('Columns, rows df1: ', df1.columns, len(df1.index))
        print('Columns, rows df2: ', df2.columns, len(df2.index))
        print('Columns, rows group result: ', group_result.columns, len(group_result.index))
        print('Similarities: ', similarities)
    return max(similarities)


def replay_merge(src1, src2, dst, key, threshold=0.95):
    inner_score = 0.0
    left_score = 0.0
    right_score = 0.0

    try:
        size = merge_size(src1, src2, key)
        if size / len(dst.index) > 10:
            logger.debug(f'Expected Join Size: {size}, destination size: {len(dst.index)}, cancelling replay.')
            return 0.0

        inner_result = src1.merge(src2, on=key)
        inner_score = ppo.compute_all_ppo(dst, inner_result, None, ppo='jaccard')['jaccard']
        if inner_score < threshold:
            left_result = src1.merge(src2, on=key, how='left')
            left_score = ppo.compute_all_ppo(dst, left_result, None, ppo='jaccard')['jaccard']
            if left_score < threshold:
                right_result = src1.merge(src2, on=key, how='right')
                right_score = ppo.compute_all_ppo(dst, right_result, None, ppo='jaccard')['jaccard']

    except Exception as e:
        logger.debug(f'Merge raised exception: {e}')
        return 0.0

    return max([inner_score, left_score, right_score])


def tiebreak_by_col_level(df_dict, pairlist):
    max_col_candidate = None
    max_col_score = 0.0

    scores_list = []

    for src, dst, score in pairlist:
        col_score = ppo.compute_col_jaccard_DF(df_dict[src], df_dict[dst])

        scores_list.append((src, dst, col_score))

        if col_score >= max_col_score:
            max_col_candidate = (src, dst, col_score)
            max_col_score = col_score

    score_dict = clustering.generate_score_dict(scores_list)

    if len(score_dict[max_col_score]) > 1:
        print("Multiple Column-Level candidates:", score_dict[max_col_score])
        s_edge_list = sorted(score_dict[max_col_score], key=hash_edge)
        print('Sorted & Hash Values:', s_edge_list, list(map(hash_edge, s_edge_list)))
        return s_edge_list[0]

    return max_col_candidate


def merge_size(left_frame, right_frame, group_by, how='inner'):
    left_groups = left_frame.groupby(group_by).size()
    right_groups = right_frame.groupby(group_by).size()
    left_keys = set(left_groups.index)
    right_keys = set(right_groups.index)
    intersection = right_keys & left_keys
    left_diff = left_keys - intersection
    right_diff = right_keys - intersection

    left_nan = len(left_frame[left_frame[group_by] != left_frame[group_by]])
    right_nan = len(right_frame[right_frame[group_by] != right_frame[group_by]])
    left_nan = 1 if left_nan == 0 and right_nan != 0 else left_nan
    right_nan = 1 if right_nan == 0 and left_nan != 0 else right_nan

    sizes = [(left_groups[group_name] * right_groups[group_name]) for group_name in intersection]
    sizes += [left_nan * right_nan]

    left_size = [left_groups[group_name] for group_name in left_diff]
    right_size = [right_groups[group_name] for group_name in right_diff]
    if how == 'inner':
        return sum(sizes)
    elif how == 'left':
        return sum(sizes + left_size)
    elif how == 'right':
        return sum(sizes + right_size)
    return sum(sizes + left_size + right_size)
