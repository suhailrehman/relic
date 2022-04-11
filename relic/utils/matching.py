import networkx as nx
import numpy as np
import copy
import string

from fuzzywuzzy import fuzz

from relic.distance.set_functions import set_jaccard_similarity
from itertools import combinations

import logging
from collections import defaultdict

from valentine import valentine_match
from valentine.algorithms import SimilarityFlooding
from valentine.metrics.metrics import one_to_one_matches


_UNIQUE_DICTIONARY = string.ascii_letters+string.digits


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def fuzzy_column_match(df1, df2):
    # print('Performing Fuzzy Match')
    left_side = {col: set(df1[col].values) for col in df1}
    right_side = {col: set(df2[col].values) for col in df2}

    left_mismatch = set(left_side.keys()) - set(right_side.keys())
    right_mismatch = set(right_side.keys()) - set(left_side.keys())

    if left_mismatch and right_mismatch:
        g = nx.DiGraph()
        for l in left_mismatch:
            for r in right_mismatch:
                logger.debug(l, r, fuzz.ratio(l, r))
                g.add_edge(l, r, weight=fuzz.WRatio(l, r))
        try:
            matching = nx.max_weight_matching(g)
            ordered_match = {(l if l in df1 else r): (r if l in df1 else l) for l, r in matching}

            df1 = df1.rename(columns=ordered_match)
        except AssertionError as e:
            logger.warning('Could not perform column fuzzy matching:', e)
            pass
    return df1, df2


def find_best_index_match(df1, index):
    col_sets = {col: set_jaccard_similarity(index, set(df1[col].values)) for col in df1}
    return sorted(col_sets.items(), key=lambda x: x[1], reverse=True)[0]


def set_df_indices(df1, df2, indexing_threshold=0.5):
    index1 = set(df1.index.values)
    index2 = set(df2.index.values)

    # print("Checking Indices")
    if set_jaccard_similarity(index1, index2) < indexing_threshold:
        # print("")
        df2_col, value1 = find_best_index_match(df2, index1)
        df1_col, value2 = find_best_index_match(df1, index2)
        if value1 < indexing_threshold and value2 < indexing_threshold:
            # TODO: Check if index is autonumbered and do something else
            # print("Resetting both indices")
            df1 = df1.reset_index()
            df2 = df2.reset_index()
        elif value1 > value2:
            # print("Resetting DF2 index to ", df2_col, value1)
            df2 = df2.set_index(df2_col, inplace=False)
        else:
            # print("Resetting DF1 index to ", df1_col, value2)
            df1 = df1.set_index(df1_col, inplace=False)
    return df1, df2


def get_common_cols(df1, df2):
    df1_cols = set(df1)
    df2_cols = set(df2)
    return df1_cols.intersection(df2_cols)


# Generates a common column lattice between dataframes df1 and df2
def generate_common_lattice(df1, df2):
    #TODO: Convert to generator expression with early exit conditions.
    df1_cols = set(df1)
    df2_cols = set(df2)

    common_cols = get_common_cols(df1, df2)
    lattice = []

    for i in range(1, len(common_cols) + 1):
        print('Lattice Generation:', i)
        level_lattice = list(combinations(common_cols, i))
        print('level:', level_lattice)
        lattice.append(level_lattice)

    # lattice = [list(itertools.combinations(common_cols, i)) for i in range(1,len(common_cols)+1)]

    # print(lattice)
    return lattice


# Schema Perturbation Function
def generate_prefix(symbol_dict: str=_UNIQUE_DICTIONARY, size: int=5) -> str:
    return ''.join(np.random.choice(list(symbol_dict), size))


def perturb_string(original_string, beta, prefix_change=False):
    if prefix_change:
        return generate_prefix()+'__'+original_string.split('__')[1]
    else:
        strlen = len(original_string)
        new_string=list(original_string)
        pertrub_idxs = np.random.randint(0,strlen, int(np.ceil(beta*strlen))).tolist()
        for ix in pertrub_idxs:
            new_string[ix] = np.random.choice(list(_UNIQUE_DICTIONARY),1)[0]
        return "".join(new_string)


def find_nodes_with_op_ancestor(gt_graph, op_list = ['pivot']):
    # Generate reverse graph
    rev_graph = nx.reverse(gt_graph)
    all_ancestors = [x for x in rev_graph.nodes() if rev_graph.out_degree(x)==0]
    dont_perturb = set()

    for u in rev_graph.nodes():
        if u not in all_ancestors:
            for v in rev_graph.nodes():
                try:
                    walk_path = nx.shortest_path(rev_graph, u, v)
                    pathGraph = nx.path_graph(walk_path)
                    path_ops = [gt_graph[v][u]['operation'] for u,v in pathGraph.edges()]
                    if set(op_list).intersection(set(path_ops)):
                        dont_perturb.add(u)
                except nx.exception.NetworkXNoPath as e:
                    pass
    return dont_perturb


def perturb_schema_dataset(df_dict, gt_graph, alpha=0.5, prefix_change=False, beta=0.2, gamma=0.4):

    # Select artifacts that do not have a pivot ancestry
    have_pivot_ancestry = find_nodes_with_op_ancestor(gt_graph)
    ok_to_perturb = set(df_dict.keys()).difference(have_pivot_ancestry)

    # Generate inverse schema map from df dict for dfs that are ok:
    inv_col_map = defaultdict(list)
    for df_label, df in df_dict.items():
        if df_label in ok_to_perturb:
            for col in df.columns:
                inv_col_map[col].append(df_label)

    # Filter by columns that have atleast two dfs attached
    inv_col_map = {k:v for k,v in inv_col_map.items() if len(v)>=2}

    # Select alpha ratio of columns to perturb - only columns that are present in more than one artifact.
    num_to_perturb = max(1,int(np.ceil(len(inv_col_map) * alpha)))
    columns_to_perturb = np.random.choice(list(inv_col_map.keys()), num_to_perturb, replace=False).tolist()

    renamed_df_dict = copy.deepcopy(df_dict)
    rename_map = defaultdict(dict)

    # Generate randint [1,gamma] versions of each column to be changed
    for col in columns_to_perturb:
        dfs_with_col = inv_col_map[col]
        num_to_perturb = np.random.randint(1, max(2,int(np.ceil(len(dfs_with_col) * gamma))))
        dfs_to_perturb = np.random.choice(dfs_with_col, num_to_perturb).tolist()

        new_col_label = perturb_string(col, beta=beta, prefix_change=prefix_change)

        logger.info(f'Modifying: {col}--->{new_col_label} in {dfs_to_perturb}')

        for df_label in dfs_to_perturb:
            renamed_df_dict[df_label] = renamed_df_dict[df_label].rename(columns={col: new_col_label})
            rename_map[df_label].update({col: new_col_label})

    return renamed_df_dict, rename_map


def schema_match_df_combo(tup, r_df_dict, already_matched=False):
    unmatched_cols = set(r_df_dict[tup[0]]).symmetric_difference(set(r_df_dict[tup[1]]))
    matches = {}

    if unmatched_cols:
        matcher = SimilarityFlooding()
        matches = one_to_one_matches(valentine_match(r_df_dict[tup[0]], r_df_dict[tup[1]], matcher, tup[0], tup[1]))

        for ((src_df_label, src_col_label), (dest_df_label, dest_col_label)) in matches.keys():
            if src_col_label != dest_col_label:
                logger.debug(f'Renaming {dest_df_label}[{dest_col_label}] --> {src_col_label}')
                r_df_dict[dest_df_label] = r_df_dict[dest_df_label].rename(columns={dest_col_label: src_col_label})

    #logger.info(f"{tup} Matches: {matches}")

    return r_df_dict, matches


def schema_match_df_triple(tup, r_df_dict, already_matched=False):
    join_dest = None
    max_col_size = 0
    for t in tup:
        if len(set(r_df_dict[t])) > max_col_size:
            join_dest = t
            max_col_size = len(set(r_df_dict[t]))

    join_sources = tuple(t for t in tup if t != join_dest)

    unmatched_cols = set(r_df_dict[join_sources[0]]).symmetric_difference(set(r_df_dict[join_dest]))
    unmatched_cols.union(set(r_df_dict[join_sources[1]]).symmetric_difference(set(r_df_dict[join_dest])))

    matches_1 = {}

    if unmatched_cols:
        matcher = SimilarityFlooding()
        matches_1 = one_to_one_matches(valentine_match(r_df_dict[join_sources[0]],
                                                       r_df_dict[join_dest], matcher,
                                                       join_sources[0], join_dest))

        for ((src_df_label, src_col_label), (dest_df_label, dest_col_label)) in matches_1.keys():
            if src_col_label != dest_col_label:
                r_df_dict[src_df_label] = r_df_dict[src_df_label].rename(columns={src_col_label: dest_col_label})

        matcher = SimilarityFlooding()
        matches_2 = one_to_one_matches(valentine_match(r_df_dict[join_sources[1]],
                                                       r_df_dict[join_dest], matcher,
                                                       join_sources[1], join_dest))

        for ((src_df_label, src_col_label), (dest_df_label, dest_col_label)) in matches_2.keys():
            if src_col_label != dest_col_label:
                r_df_dict[src_df_label] = r_df_dict[src_df_label].rename(columns={src_col_label: dest_col_label})

        matches_1.update(matches_2)

    return r_df_dict, matches_1
