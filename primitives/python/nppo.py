#!/usr/bin/env python

"""nppo.py: Detectors for Non-Point Preserving Operations."""

__author__      = "Suhail Rehman"
__email__       = "suhail@uchicago.edu"

import pandas as pd
import numpy as np
import itertools
import glob
import os
from dfs import *

from collections import defaultdict

import clustering
import csv

import pprint

def get_common_cols(df1,df2):
    df1_cols = set(df1)
    df2_cols = set(df2)
    return df1_cols.intersection(df2_cols)

# Join Detection
# Generates a common column lattice between dataframes df1 and df2
def generate_common_lattice(df1,df2):
    df1_cols = set(df1)
    df2_cols = set(df2)

    common_cols = get_common_cols(df1,df2)
    lattice = []

    for i in range(1,len(common_cols)+1):
        print('Lattice Generation:', i)
        level_lattice = list(itertools.combinations(common_cols,i))
        print('level:' , level_lattice)
        lattice.append(level_lattice)

    #lattice = [list(itertools.combinations(common_cols, i)) for i in range(1,len(common_cols)+1)]

    #print(lattice)
    return lattice


def get_max_coherent_columns_1(df1, df2):
    common_cols = get_common_cols(df1,df2)

    coherent_1_cols = set(col for col in common_cols if check_col_containment(df1,df2,col))

    if coherent_1_cols:
        if check_col_group_containment(df1, df2, coherent_1_cols):
            return coherent_1_cols

    return None


# Looks for perfect column containment of colname between dataframes df1, df2
def check_col_containment(df1, df2, colname, col2name=None):
    if(col2name==None):
        col2name = colname
    return set(df1[colname]).issubset(set(df2[col2name]))

# Looks for perfect colgroup containment of colgroup between dataframes df1, df2
def check_col_group_containment(df1, df2, colgroup, colgroup2=None):
    if(colgroup2==None):
        colgroup2 = colgroup

    df1valset = set(frozenset(u) for u in df1[list(colgroup)].values.tolist())
    df2valset = set(frozenset(u) for u in df2[list(colgroup2)].values.tolist())

    #print(df2valset)

    return df1valset.issubset(df2valset)

# Removes all supersets of badtip from lattice
def remove_tup_lattice(lattice, badtup):
    # TODO: start comparing from len(badtup) level upwards
    for i in range(len(lattice)):
        level = lattice[i]
        new_level = [item for item in level if not set(badtup).issubset(set(item))]
        lattice[i] = new_level
    #print(lattice)
    return lattice

# Check for df1 >= df2 and max columns contained therein
def get_max_coherent_columns(df1,df2):
    lattice = generate_common_lattice(df1,df2)

    # All common columns are coherent at start
    coherent_cols = set(itertools.chain(*lattice[0]))

    for i in range(len(lattice)):
        print('Checking lattice level', i)
        level = lattice[i]
        new_lattice = lattice
        for tup in level:
            print('Checking:', tup)
            contained = check_col_group_containment(df1,df2,tup)
            if not contained:
                print('removing', tup)
                new_lattice = remove_tup_lattice(new_lattice, tup)
        lattice = new_lattice


    non_empty = [l for l in lattice if len(l)>0]
    if non_empty:
        return  non_empty[-1][0]
    return []


# Check for df1 >= df2 and max columns contained therein
def get_max_coherent_columns_skip(df1,df2):
    common_cols = get_common_cols(df1,df2)

    for i in range(len(lattice)):
        print('Checking lattice level', i)
        level = lattice[i]
        new_lattice = lattice
        for tup in level:
            print('Checking:', tup)
            contained = check_col_group_containment(df1,df2,tup)
            if not contained:
                print('removing', tup)
                new_lattice = remove_tup_lattice(new_lattice, tup)
        lattice = new_lattice


    non_empty = [l for l in lattice if len(l)>0]
    if non_empty:
        return  non_empty[-1][0]
    return []

def evaluate_join_triple(combo, df_dict):
    print(combo)
    sizes = {x: len(set(df_dict[x])) for x in combo}
    if max(sizes.values())==min(sizes.values()):
        return None
    join_dest = list(sizes.keys())[list(sizes.values()).index(max(sizes.values()))]
    join_sources = tuple(x for x in combo if x is not join_dest)

    if set(df_dict[join_sources[0]]).union(set(df_dict[join_sources[1]])) == set(df_dict[join_dest]):
        print ('Column Union Match:', join_dest, join_sources)
        print('Checking column coherency of', join_sources[0], join_dest)
        coherent_1 = get_max_coherent_columns_1(df_dict[join_sources[0]], df_dict[join_dest])
        print('Checking column coherency of', join_sources[1], join_dest)
        coherent_2 = get_max_coherent_columns_1(df_dict[join_sources[1]], df_dict[join_dest])

        # Check if the coherent columns generate the output set
        if set(coherent_1).union(set(coherent_2)) == set(df_dict[join_dest]):
            print('coherent:', (join_dest, join_sources))
            if set(coherent_1).intersection(set(coherent_2)): # Check if the intersection is not null
                print('intersection: ', (join_dest, join_sources))
                return (join_dest, join_sources)
    return None


def build_df_dict(csvdir):
    artifacts = [os.path.basename(p) for p in glob.glob(csvdir+'*.csv')]
    return {artifact: get_dataframe(csvdir, artifact) for artifact in artifacts}


# Given a Notebook (nb_name) and directory full of csvs (dir), return all probable joins via containment
def get_all_joins_wf(nb_name, csvdir):
    joins = []
    artifact_dir = csvdir
    artifacts = [os.path.basename(p) for p in glob.glob(artifact_dir+'*.csv')]
    df_dict = {artifact: get_dataframe(csvdir, artifact) for artifact in artifacts}
    combos = itertools.combinations(df_dict.keys(),3)
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
    #if len(col1) == len(col2):
    #    return False
    src, dst = ((col1,col2) if len(col1) > len(col2) else (col2, col1))
    if(set(src.values) == set(dst.values)): # Check set intersection
        if(len(set(src.values)) == len(dst.values)): # Check if destination set is unique and fully contained
            return True
    else:
        return False

#TODO: Column-Lattice based groupby checks

def df_groupby_check(df1,df2):
    combinations = itertools.product(list(df1), list(df2))
    for col1,col2 in combinations:
        if(column_groupby_check(df1[col1], df2[col2])):
            return str(col1), str(col2)
    return False

def get_all_groupbys_wf(nb_name, csvdir):
    artifact_dir = csvdir
    artifacts = [os.path.basename(p) for p in glob.glob(artifact_dir+'*.csv')]
    df_dict = {artifact: get_dataframe(csvdir, artifact) for artifact in artifacts}
    combinations = itertools.combinations(df_dict.keys(),2)
    for df1, df2 in combinations:
        result = df_groupby_check(df_dict[df1], df_dict[df2])
        if result:
            print(df1, result[0], df2, result[1])
    return True


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
    schema_combos = [combo for combo in itertools.combinations(clusters.keys(),3)]
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
        #print(max_common_col)
        pruned_candidates.append(join_schemas[join_result][max_common_col])

    return pruned_candidates




def enumerate_join_candidates_new(join_schemas, clusters, df_dict):
    candidates = {}

    for schema in join_schemas:
        max_join_union_size = 0
        join_candidates = defaultdict(list)
        #print(schema)
        join_l, join_r = clusters[schema[0][0][0]], clusters[schema[0][0][1]]
        join_dest = clusters[schema[0][1]]
        for jl in join_l:
            for jr in join_r:
                for jd in join_dest:
                    coherent_1 = simple_coherency_check(df_dict[jd],df_dict[jl])
                    coherent_2 = simple_coherency_check(df_dict[jd],df_dict[jr])
                    # Check if the coherent columns generate the output set
                    #print(coherent_1, coherent_2)
                    if coherent_1 and coherent_2:
                        union = set(coherent_1).union(set(coherent_2))
                        size = len(union.intersection(set(df_dict[jd])))
                        if size > 0 and size >= max_join_union_size:
                            if set(coherent_1).intersection(set(coherent_2)): # Check if the intersection is not null
                                join_candidates[size].append((jl,jr,jd))
                                max_join_union_size = size

        candidates[schema[0]] = join_candidates[max_join_union_size]
    return candidates



def check_minimal_extra_values(candidates, df_dict):
    # For each schema pair, check for values that should have been joined but are not
    # present
    best_join_candidates = {}
    for schema, candidate_list in candidates.items():
        #Set Difference
        best_matches = defaultdict(list)
        least_surplus = np.inf
        for combo in candidate_list:
            jl,jr,jd = combo
            coherent_l = simple_coherency_check(df_dict[jd],df_dict[jl])
            coherent_r = simple_coherency_check(df_dict[jd],df_dict[jr])

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
                best_matches[total_excess] = [(jl,jr,jd)]
                least_surplus = total_excess

        best_join_candidates[schema] = best_matches[least_surplus]

    return best_join_candidates

def write_join_candidates(join_candidate_list, filename):
    with open(filename,'w') as fp:
        csv_out = csv.writer(fp)
        for row in join_candidate_list:
            csv_out.writerow(row)


def find_all_joins_df_dict(df_dict):
    clusters= clustering.exact_schema_cluster(df_dict)
    print(len(clusters), " schema clusters total")
    join_schemas = find_join_schemas_maximal(clusters)
    print(len(join_schemas), " joinable schema combinations")
    pruned = prune_join_schemas(join_schemas)
    #print(join_schemas, " joinable schema combinations after pruning")

    jc = enumerate_join_candidates_new(pruned, clusters, df_dict)
    #print("Schema Candidates:\n")
    #pp = pprint.PrettyPrinter(indent=1)
    #pp.pprint(jc)
    jc_pruned = {k:v for k,v in jc.items() if len(v) > 0}
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
