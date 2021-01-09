#!/usr/bin/env python

"""nppo.py: Detectors for Non-Point Preserving Operations."""

__author__ = "Suhail Rehman"
__email__ = "suhail@uchicago.edu"

import pandas as pd
import numpy as np
import itertools
import glob
import os

import networkx as nx

from dfs import *

from collections import defaultdict

from lineage import similarity

import clustering
import csv

import pprint
import operator

from tqdm.auto import tqdm
from hashlib import md5

import math


def hash_edge(x):
    w = "+".join(sorted(x[:-1])).encode('utf8')
    return md5(w).hexdigest()

def hash_edge_join(x):
    w = "+".join((x[0][0], x[0][1], x[1])).encode('utf8')
    return md5(w).hexdigest()


def get_common_cols(df1, df2):
    df1_cols = set(df1)
    df2_cols = set(df2)
    return df1_cols.intersection(df2_cols)


# Join Detection
# Generates a common column lattice between dataframes df1 and df2
def generate_common_lattice(df1, df2):
    df1_cols = set(df1)
    df2_cols = set(df2)

    common_cols = get_common_cols(df1, df2)
    lattice = []

    for i in range(1, len(common_cols) + 1):
        print('Lattice Generation:', i)
        level_lattice = list(itertools.combinations(common_cols, i))
        print('level:', level_lattice)
        lattice.append(level_lattice)

    # lattice = [list(itertools.combinations(common_cols, i)) for i in range(1,len(common_cols)+1)]

    # print(lattice)
    return lattice


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



# Removes all supersets of badtip from lattice
def remove_tup_lattice(lattice, badtup):
    # TODO: start comparing from len(badtup) level upwards
    for i in range(len(lattice)):
        level = lattice[i]
        new_level = [item for item in level if not set(badtup).issubset(set(item))]
        lattice[i] = new_level
    # print(lattice)
    return lattice


# Check for df1 >= df2 and max columns contained therein
def get_max_coherent_columns(df1, df2):
    lattice = generate_common_lattice(df1, df2)

    # All common columns are coherent at start
    coherent_cols = set(itertools.chain(*lattice[0]))

    for i in range(len(lattice)):
        print('Checking lattice level', i)
        level = lattice[i]
        new_lattice = lattice
        for tup in level:
            print('Checking:', tup)
            contained = check_col_group_containment(df1, df2, tup)
            if not contained:
                print('removing', tup)
                new_lattice = remove_tup_lattice(new_lattice, tup)
        lattice = new_lattice

    non_empty = [l for l in lattice if len(l) > 0]
    if non_empty:
        return non_empty[-1][0]
    return []


# Check for df1 >= df2 and max columns contained therein
def get_max_coherent_columns_skip(df1, df2):
    common_cols = get_common_cols(df1, df2)

    for i in range(len(lattice)):
        print('Checking lattice level', i)
        level = lattice[i]
        new_lattice = lattice
        for tup in level:
            print('Checking:', tup)
            contained = check_col_group_containment(df1, df2, tup)
            if not contained:
                print('removing', tup)
                new_lattice = remove_tup_lattice(new_lattice, tup)
        lattice = new_lattice

    non_empty = [l for l in lattice if len(l) > 0]
    if non_empty:
        return non_empty[-1][0]
    return []


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


def evaluate_join_triple(combo, df_dict):
    print(combo)
    sizes = {x: len(set(df_dict[x])) for x in combo}
    if max(sizes.values()) == min(sizes.values()):
        return None
    join_dest = list(sizes.keys())[list(sizes.values()).index(max(sizes.values()))]
    join_sources = tuple(x for x in combo if x is not join_dest)

    if set(df_dict[join_sources[0]]).union(set(df_dict[join_sources[1]])) == set(df_dict[join_dest]):
        print('Column Union Match:', join_dest, join_sources)
        print('Checking column coherency of', join_sources[0], join_dest)
        coherent_1 = get_max_coherent_columns_1(df_dict[join_sources[0]], df_dict[join_dest])
        print('Checking column coherency of', join_sources[1], join_dest)
        coherent_2 = get_max_coherent_columns_1(df_dict[join_sources[1]], df_dict[join_dest])

        # Check if the coherent columns generate the output set
        if set(coherent_1).union(set(coherent_2)) == set(df_dict[join_dest]):
            print('coherent:', (join_dest, join_sources))
            if set(coherent_1).intersection(set(coherent_2)):  # Check if the intersection is not null
                print('intersection: ', (join_dest, join_sources))
                return (join_dest, join_sources)
    return None


def build_df_dict(csvdir):
    artifacts = [os.path.basename(p) for p in glob.glob(csvdir + '*.csv')]
    return {artifact: get_dataframe(csvdir, artifact) for artifact in artifacts}


# Given a Notebook (nb_name) and directory full of csvs (dir), return all probable joins via containment
def get_all_joins_wf(nb_name, csvdir):
    joins = []
    artifact_dir = csvdir
    artifacts = [os.path.basename(p) for p in glob.glob(artifact_dir + '*.csv')]
    df_dict = {artifact: get_dataframe(csvdir, artifact) for artifact in artifacts}
    combos = itertools.combinations(df_dict.keys(), 3)
    for combo in combos:
        result = evaluate_join_triple(combo, df_dict)
        if result:
            joins.append(result)
    return joins


## Group By Detection

# GroupBy detector test:
# Expects Pandas Series Objects
def column_groupby_check(col1, col2):
    # If the columns are same, return False
    # if len(col1) == len(col2):
    #    return False
    src, dst = ((col1, col2) if len(col1) > len(col2) else (col2, col1))

    # TODO: avoid repeated set generation

    # Keyness checking
    if (len(set(dst.values)) != len(dst.values)):
        return False

    if (len(src.values) == len(dst.values)):
        return False

    if (set(src.values) == set(dst.values)):  # Check set intersection
        if (len(set(src.values)) == len(dst.values)):  # Check if destination set is unique and fully contained
            return True
    else:
        return False


# TODO: Column-Lattice based groupby checks

def df_groupby_check(df1, df2):
    combinations = itertools.product(list(df1), list(df2))
    for col1, col2 in combinations:
        if (column_groupby_check(df1[col1], df2[col2])):
            return str(col1), str(col2)
    return False


def df_groupby_check_direct(df1, df2):
    common_cols = set(list(df1)).intersection(set(list(df2)))
    for col in common_cols:
        if (column_groupby_check(df1[col], df2[col])):
            return col, col
    return False


def get_group_agg_cols(df1, df2, contaiment_threshold=0.9, sim_threshold=0.9, lattice_check=False, debug=False):
    group_cols = []
    group_col_containment = []
    agg_cols = []

    # TODO: Use a column matching map instead
    common_cols = set(list(df1)).intersection(set(list(df2)))

    for col in common_cols:
        containment = col_containment(df1, df2, col)
        jsim = similarity.set_jaccard_similarity(set(df1[col].values), set(df2[col].values))
        if debug:
            print('Col, containment, jsim: ', col , containment, jsim)
        if containment >= contaiment_threshold and jsim >= sim_threshold:
            group_cols.append(col)
            group_col_containment.append(containment)

        else:
            agg_cols.append(col)

    if debug:
        print('Determining Group Cols: ')
        print(group_cols, group_col_containment)

    if lattice_check and group_cols:
        group_cols = explore_group_lattice(df1, df2, group_cols, jaccard_threshold=sim_threshold)

    # Find the group value containment for the contained columns:
    srcvalset = set(frozenset(u) for u in df1[group_cols].values.tolist())
    dstvalset = set(frozenset(u) for u in df2[group_cols].values.tolist())

    if debug:
        print('Source Val set:', srcvalset)
        print('Dest Val set:', dstvalset)
        print('Symmdiff : ', srcvalset.symmetric_difference(dstvalset))

    missing_vals = (len(srcvalset.symmetric_difference(dstvalset)) / len(srcvalset))

    return group_cols, agg_cols, missing_vals


def colgroup_keyness_check(df, colgroup):
    return columnset_keyness_ratio(df, colgroup) == 1.0


def df_groupby_check_new(d1, d2, df_dict, g_inferred, debug=False, strict_schema=False, lattice_check=False, null_aggs=False):
    # TODO: Column matching
    df1 = df_dict[d1]
    df2 = df_dict[d2]
    common_cols = set(list(df1)).intersection(set(list(df2)))

    # Strict Schema Check
    sym_diff_cols = set(list(df1)).symmetric_difference(set(list(df2)))
    if sym_diff_cols and strict_schema:
        if debug:
            print('Failed strict schema check')
        return 0.0

    if not common_cols:
        if debug:
            print('DFs have no common columns')
        return 0.0

    # Contraction Check:
    if len(df1.index) == len(df2.index):
        if debug:
            print('There is no len contraction')
        return 0.0

    src, dst = ((df1, df2) if len(df1.index) > len(df2.index) else (df2, df1))

    # Containment check and dividing columns into group and aggregate
    group_cols, agg_cols, missing_vals = get_group_agg_cols(src, dst, sim_threshold=1.0,
                                                            lattice_check=lattice_check, debug=debug)

    if debug:
        print('Groupcols', group_cols)
        print('agg_cols', agg_cols)
        print('missing_vals', missing_vals)

    if not group_cols:
        if debug:
            print('No group columns detected')
        return 0.0

    if not agg_cols and not null_aggs: # Don't allow aggregates to be null
        if debug:
            print('No agg cols in common between the dataframes')
        return 0.0

    if missing_vals > 0.0:
        if debug:
            print('GroupBy has missing values')
        return 0.0

    # TODO: Lattice exploration
    src_group_keyness_ratio = columnset_keyness_ratio(src, group_cols)
    dst_group_keyness_ratio = columnset_keyness_ratio(dst, group_cols)

    if src_group_keyness_ratio == 1.0:
        if debug:
            print('Source group columns are also keys, groupby unlikely', src_group_keyness_ratioc)
        return 0.0

    if dst_group_keyness_ratio < 1.0:
        if debug:
            print('Group keyness below threshold:', dst_group_keyness_ratio)
        return 0.0

    column_diff = similarity.set_jaccard_similarity(set(df1.columns), set(df2.columns))

    contraction_ratio = len(src.index) / len(dst.index)

    final_val = ((1.0 * len(group_cols) * dst_group_keyness_ratio) - missing_vals) #* column_diff
    if debug:
        print('final_val = (1.0 * len(group_cols) * group_keyness_ratio) - missing_vals') #* column_diff')
        print(final_val, ' = ', '(1.0 *', len(group_cols), '*', dst_group_keyness_ratio, ')-', missing_vals)

    return final_val  # * contraction_ratio


def df_groupby_check_direct_multi(df1, df2):
    common_cols = set(list(df1)).intersection(set(list(df2)))
    for i in range(1, 3):
        colcombos = itertools.combinations(common_cols, i)
        for col in common_cols:
            if (column_groupby_check(df1[col], df2[col])):
                return col, col
    return False


def get_all_groupbys_wf(nb_name, csvdir):
    artifact_dir = csvdir
    artifacts = [os.path.basename(p) for p in glob.glob(artifact_dir + '*.csv')]
    df_dict = {artifact: get_dataframe(csvdir, artifact) for artifact in artifacts}
    combinations = itertools.combinations(df_dict.keys(), 2)
    for df1, df2 in combinations:
        result = df_groupby_check(df_dict[df1], df_dict[df2])
        if result:
            print(df1, result[0], df2, result[1])
    return True


def get_all_groupbys_dfdict(df_dict):
    return_result = []
    combinations = itertools.combinations(df_dict.keys(), 2)
    for df1, df2 in combinations:
        result = df_groupby_check_direct(df_dict[df1], df_dict[df2])
        if result:
            return_result.append((df1, result[0], df2, result[1]))
    return return_result

'''
# Pivot Detection
# Checks if df1 is a pivot of df2 or vice versa
# df1, df2 are dataframes
def pivot_detector(df1, df2):
    df1_cols = set(df1)
    df2_cols = set(df2)

    intersect = False

    for col in df1_cols:
        intersect = set(df1[col]).intersection(df2_cols)
        if intersect:
            return col, intersect

    for col in df2_cols:
        intersect = set(df2[col]).intersection(df1_cols)
        if intersect:
            return col, intersect

    return intersect
'''

# Improved Join Detectors

def simple_coherency_check(base_df, join_dest_df):
    return get_max_coherent_columns_1(base_df, join_dest_df)


def find_join_order_general(combo):
    combo_set = set(combo)
    max_combo = None
    max_col_number = 0
    for join_dest in combo_set:
        join_sources = combo_set - set([join_dest])

        common_cols = set()
        for source in join_sources:
            if not source.intersection(join_dest):
                common_cols = None
                break
            common_cols = common_cols.union(source)

        if not common_cols:
            continue

        common_cols = common_cols.intersection(join_dest)
        if not common_cols:
            continue

        if len(common_cols) > max_col_number:
            max_col_number = len(common_cols)
            max_combo = (tuple(join_sources), join_dest)

    if not max_combo or not common_cols:
        return None
    return tuple(max_combo), len(common_cols)


def find_join_schemas_maximal(clusters):
    schema_combos = [combo for combo in itertools.combinations(clusters.keys(), 3)]
    join_schemas = defaultdict(lambda: defaultdict(list))
    for x in schema_combos:
        result = find_join_order_general(x)
        if result:
            combo, val = result
            join_result = combo[1]
            join_schemas[join_result][val].append(combo)
    return join_schemas


def prune_join_schemas(join_schemas):
    pruned_candidates = []
    for join_result in join_schemas.keys():
        max_common_col = max(join_schemas[join_result])
        # print(max_common_col)
        pruned_candidates.append(join_schemas[join_result][max_common_col])

    return pruned_candidates


def enumerate_join_candidates_new(join_schemas, clusters, df_dict):
    candidates = {}

    for schema in join_schemas:
        max_join_union_size = 0
        join_candidates = defaultdict(list)
        # print(schema)
        join_l, join_r = clusters[schema[0][0][0]], clusters[schema[0][0][1]]
        join_dest = clusters[schema[0][1]]
        for jl in join_l:
            for jr in join_r:
                for jd in join_dest:
                    coherent_1 = simple_coherency_check(df_dict[jd], df_dict[jl])
                    coherent_2 = simple_coherency_check(df_dict[jd], df_dict[jr])
                    # Check if the coherent columns generate the output set
                    # print(coherent_1, coherent_2)
                    if coherent_1 and coherent_2:
                        union = set(coherent_1).union(set(coherent_2))
                        size = len(union.intersection(set(df_dict[jd])))
                        if size > 0 and size >= max_join_union_size:
                            if set(coherent_1).intersection(set(coherent_2)):  # Check if the intersection is not null
                                join_candidates[size].append((jl, jr, jd))
                                max_join_union_size = size

        candidates[schema[0]] = join_candidates[max_join_union_size]
    return candidates


def check_minimal_extra_values(candidates, df_dict):
    # For each schema pair, check for values that should have been joined but are not
    # present
    best_join_candidates = {}
    for schema, candidate_list in candidates.items():
        # Set Difference
        best_matches = defaultdict(list)
        least_surplus = np.inf
        for combo in candidate_list:
            jl, jr, jd = combo
            coherent_l = simple_coherency_check(df_dict[jd], df_dict[jl])
            coherent_r = simple_coherency_check(df_dict[jd], df_dict[jr])

            jlvalset = set(frozenset(u) for u in df_dict[jl][list(coherent_l)].values.tolist())
            jrvalset = set(frozenset(u) for u in df_dict[jr][list(coherent_r)].values.tolist())
            jdlvalset = set(frozenset(u) for u in df_dict[jd][list(coherent_l)].values.tolist())
            jdrvalset = set(frozenset(u) for u in df_dict[jd][list(coherent_r)].values.tolist())

            left_size = len(jlvalset - jdlvalset)
            right_size = len(jrvalset - jdrvalset)

            total_excess = left_size + right_size

            if total_excess < least_surplus:
                print(combo)
                print(total_excess)
                best_matches[total_excess] = [(jl, jr, jd)]
                least_surplus = total_excess

        best_join_candidates[schema] = best_matches[least_surplus]

    return best_join_candidates


def write_join_candidates(join_candidate_list, filename):
    with open(filename, 'w') as fp:
        csv_out = csv.writer(fp)
        for row in join_candidate_list:
            csv_out.writerow(row)


def find_all_joins_df_dict(df_dict):
    clusters = clustering.exact_schema_cluster(df_dict)
    print(len(clusters), " schema clusters total")
    join_schemas = find_join_schemas_maximal(clusters)
    print(len(join_schemas), " joinable schema combinations")
    pruned = prune_join_schemas(join_schemas)
    print(join_schemas, " joinable schema combinations after pruning")

    jc = enumerate_join_candidates_new(pruned, clusters, df_dict)
    print("Schema Candidates:\n")
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint(jc)
    jc_pruned = {k: v for k, v in jc.items() if len(v) > 0}
    best_result = check_minimal_extra_values(jc_pruned, df_dict)
    final_list = []
    for val in best_result.values():
        final_list.extend(val)

    return final_list


def add_join_edges(join_list, G):
    for join in join_list:
        G.add_edge(join[0], join[2], weight=0)
        G.add_edge(join[1], join[2], weight=0)
    return G


def add_group_edges(group_list, G):
    for group in group_list:
        G.add_edge(group[0], group[1], weight=0)
    return G


# New Join Check Implementation
def df_join_check_new(df1, df2, df_dict, g_inferred):
    # TODO: Column matching

    schema_scores = {}
    for df3 in g_inferred.neighbors(df1):
        schema_scores.update(score_join_schema(df1, df2, df3, df_dict))
    for df3 in g_inferred.neighbors(df2):
        schema_scores.update(score_join_schema(df1, df2, df3, df_dict))

    # print(schema_scores)

    if schema_scores:
        return max(schema_scores.keys())
    else:
        return 0.0


def score_join_schema(df1, df2, df3, df_dict, debug=False, replay=True):
    if debug:
        print("Join Scoring:", df1, df2, df3)
    combo_set = set((df1, df2, df3))
    max_combo = None
    max_col_number = 0

    columns_dict = {df: set(df_dict[df].columns) for df in combo_set}

    # First Check for at least one common column in triple to act as key
    common_cols = columns_dict[df1].intersection(columns_dict[df2]).intersection(columns_dict[df3])

    if not common_cols:
        if debug:
            print('No common cols')
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

        jaccard = similarity.set_jaccard_similarity(columns_dict[join_dest], column_union) * len(
            columns_dict[join_dest])

        if debug:
            print('DEBUG', source, other_source, join_dest, symm_diff, column_union, jaccard)

        if jaccard > max_col_number:
            max_col_number = jaccard
            max_combo = (tuple(join_sources), join_dest)

    if not max_combo or not common_cols:
        if debug:
            print('No Max combo')
        return ((df1, df2), df3, 0.0)

    df1, df2 = max_combo[0]
    df3 = max_combo[1]

    df1_columns = columns_dict[df1].intersection(columns_dict[df3]) - columns_dict[df2]
    df2_columns = columns_dict[df2].intersection(columns_dict[df3]) - columns_dict[df1]

    # Check for non-key contributions from either side:
    if not df1_columns or not df2_columns:
        if debug:
            print('Poor contribution from either side')
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

        if debug:
            print('Join Key:', join_key, 'Containment', containments[join_key])
            # print('Key List:', key_list)
            # print('df1_merge_vals', df1_merge_vals)
            # print('df2_merge_vales', df2_merge_vals)

    # print(df1_containment, df2_containment)

    max_containment = max(containments.items(), key=operator.itemgetter(1))[0]

    if containments[max_containment] < 0.001:
        if debug:
            print('Poor containment')
        return ((df1, df2), df3, 0.0)

    if debug:
        print('df1,df2,df3:', df1, df2, df3)
        print('column_sets:', df1_columns, df2_columns, columns_dict[df3])
        print('join key:', max_containment)



    df1_containment = col_group_containment(df_dict[df1], df_dict[df3], df1_columns)
    df2_containment = col_group_containment(df_dict[df2], df_dict[df3], df2_columns)

    if df1_containment < 0.01 or df2_containment < 0.01:
        if debug:
            print("poor single-sided containments")
        return ((df1, df2), df3, 0.0)

    # TODO: Check coherency here
    # Do Join replay for now:
    if replay:
        try:
            replay_score1 = replay_merge(df_dict[df1], df_dict[df2], df_dict[df3], max_containment)
            replay_score2 = replay_merge(df_dict[df2], df_dict[df1], df_dict[df3], max_containment)
            replay_score = max(replay_score1, replay_score2)
        except (ValueError, pd.errors.MergeError) as e:
            print('Could not merge ', df1, df2, 'to produce', df3, 'using', max_containment)
            replay_score = 0.0
            pass
    else:
        replay_score = df1_containment * df2_containment

    if debug:
        # print('max_col_ratio, df1contain, df2contain', max_col_number, df1_containment, df2_containment)
        print('max_col_ratio, maxcontain, replay_score', max_col_number, containments[max_containment], replay_score)

    return (max_combo[0], max_combo[1], replay_score)


#### Complete Groupby implementation

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

    index_ratio = similarity.set_jaccard_similarity(set(df1.index), set(df2.index))
    column_ratio = similarity.set_max_containment(set(df1.columns), set(df2.columns))
    cell_score = 1.0 - similarity.compute_jaccard_DF(df1, df2)

    # print(index_ratio, column_ratio, cell_score)

    return index_ratio * column_ratio * cell_score


### Common NPPO component search routine:


def find_components_nppo_edge(g_inferred, df_dict, edge_num, nppo_dict, nppo_function=df_groupby_check_new,
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
        return None, edge_num, nppo_dict, None, all_cmp_pairs_similarties

    # TODO: Common scoring function. right now 1.0 for all edges found.
    all_cmp_pairs_similarties.sort(key=lambda x: x[2], reverse=True)

    # print('NNPOs detected')
    # print(all_cmp_pairs_similarties)

    score_dict = clustering.generate_score_dict(all_cmp_pairs_similarties)
    maxscore = max(score_dict)

    if maxscore < threshold:
        print('No more edges above threshold')
        return None, edge_num, nppo_dict, None, all_cmp_pairs_similarties

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

        if nppo_function == df_groupby_check_new:
            #print("Breaking tie by column-level", score_dict[maxscore])
            #src, dst, score = tiebreak_by_col_level(df_dict, score_dict[maxscore])
            print("Breaking tie by groupby-replay", score_dict[maxscore])
            src, dst, score = tiebreak_by_groupby_replay(df_dict, score_dict[maxscore])
        else:
            print("Breaking tie by contraction ratio", score_dict[maxscore])
            src, dst, score = tie_break_by_contraction_ratio(df_dict, score_dict[maxscore])

    else:
        src, dst, score = score_dict[maxscore][0]

    print('Adding nppo/group edge', src, dst, score)
    g_inferred.add_edge(src, dst, weight=score, num=edge_num, type=label)
    edge_num += 1
    return g_inferred, edge_num, nppo_dict, (src, dst), all_cmp_pairs_similarties


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

    #print('Raw triples: ', len(new_edges))
    #union = tuples.union(set(new_edges))
    #print('After union: ', len(union))
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
            similarities = score_join_schema(d1, d2, d3, df_dict)
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
        score = similarity.set_jaccard_similarity(valueset, col_data)
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

    if debug:
        print('df1index, df2index, df1columns, df2columns', df1.index.name, df2.index.name, df1.columns, df2.columns)

    if schema_check: # Check for at most one common column (index)
        common_cols = set(df1).intersection(set(df2))
        if len(common_cols) > 1:
            return 0.0

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
        index_containment = similarity.set_jaccard_similarity(set(src[index_name_match].values), set(dst.index.values))
        if debug:
            print("Index name match & containment:", index_name_match, index_containment)

    # otherwise use containment to figure out src and dst:
    else:
        return 0.0
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
        return 0.0

    # Check column-value containment
    try:
        coldict = find_column_set_match(src, set(dst), string_convert=True)
        max_col_score = max(coldict.keys())
        if debug:
            print('Column match dict: ', coldict)
    except ValueError as e:
        return 0.0

    # print(index_containment, max_col_score)
    if max_col_score <= col_match_threshold:
        return 0.0

    # Check pivot value containment if flag is set:
    if match_values:
        flat_values = dst.values.flatten()

        if flat_values.dtype == 'float64':
            valdict = find_column_set_match(src, set([x for x in flat_values if ~np.isnan(x)]))
        else:
            valdict = find_column_set_match(src, set([x for x in flat_values]))

        max_val_score = max(valdict.keys())
        if debug:
            print('Value column match: ', valdict)
            print('Pivot scores:', index_containment, max_col_score, max_val_score)

        #Pivot replay section
        #index: src->index_name_match
        #col: coldict['max_col_score']
        #val: valdict['max_val_Score']
        if (index_containment * max_col_score) + max_val_score > replay_threshold:
            similarities = []
            for col in coldict[max_col_score]:
                for val in valdict[max_val_score]:
                    try:
                        replaydf = src.pivot_table(index=index_name_match, columns=col, values=val, aggfunc=max)
                        similarities.append(similarity.compute_jaccard_DF(dst, replaydf))
                    except Exception as e:
                        print('Cannot pivot:', df1_name, df2_name, index_name_match, col, val, e)
                        similarities.append(0.0)
            return max(similarities)

    return index_containment * max_col_score


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
        s_edge_list = sorted(score_dict[min_dff], key=hash_edge)
        return s_edge_list[0]

    return min_ts_diff


def tiebreak_by_groupby_replay(df_dict, pairlist):
    max_replay_candidate = None
    max_cell_score = 0.0

    scores_list = []

    for src, dst, score in pairlist:
        replay_score = replay_groupby(df_dict[src], df_dict[dst])

        scores_list.append((src, dst, replay_score))

        if replay_score >= max_cell_score:
            max_replay_candidate = (src, dst, replay_score)
            max_cell_score = replay_score

    score_dict = clustering.generate_score_dict(scores_list)

    if len(score_dict[max_cell_score]) > 1:
        print("Multiple Replay candidates:", score_dict[max_cell_score])
        s_edge_list = sorted(score_dict[max_cell_score], key=hash_edge)
        return s_edge_list[0]

    return max_replay_candidate


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

    group_object = src.groupby(grouplist)
    for agg_op in agg_ops:
        try:
            agg_func = getattr(group_object, agg_op)
            group_result = agg_func().reset_index()
            similarities.append(similarity.compute_jaccard_DF(dst, group_result))
        except Exception as e:
            print('Warning GroupBy op failed: ', agg_op, e)
            similarities.append(0.0)

    if debug:
        print('GroupBy performed on column: ', grouplist)
        print('Columns, rows df1: ', df1.columns, len(df1.index))
        print('Columns, rows df2: ', df2.columns, len(df2.index))
        #print('Columns, rows group result: ', group_result.columns, len(group_result.index))
        print('Similarities: ', similarities)
    return max(similarities)


def replay_merge(src1, src2, dst, key, threshold=0.95):
    inner_score = 0.0
    left_score = 0.0
    right_score = 0.0

    try:
        size = merge_size(src1, src2, key)
        if size / len(dst.index) > 10:
            return 0.0

        inner_result = src1.merge(src2, on=key)
        inner_score = similarity.compute_jaccard_DF(dst, inner_result, containment=False)
        if inner_score < threshold:
            left_result = src1.merge(src2, on=key, how='left')
            left_score = similarity.compute_jaccard_DF(dst, left_result, containment=False)
            if left_score < threshold:
                right_result = src1.merge(src2, on=key, how='right')
                right_score = similarity.compute_jaccard_DF(dst, right_result, containment=False)

    except Exception as e:
        #raise
        return 0.0

    return max([inner_score, left_score, right_score])


def tiebreak_by_col_level(df_dict, pairlist):
    max_col_candidate = None
    max_col_score = 0.0

    scores_list = []

    for src, dst, score in pairlist:
        col_score = similarity.compute_col_jaccard_DF(df_dict[src], df_dict[dst])

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
