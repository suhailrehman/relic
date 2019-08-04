#!/usr/bin/env python

"""nppo.py: Detectors for Non-Point Preserving Operations."""

__author__      = "Suhail Rehman"
__email__       = "suhail@uchicago.edu"

import pandas as pd
import itertools

# Join Detection
# Generates a common column lattice between dataframes df1 and df2
def generate_common_lattice(df1,df2):
    df1_cols = set(df1)
    df2_cols = set(df2)

    common_cols = df1_cols.intersection(df2_cols)
    #print(common_cols)

    lattice = [list(itertools.combinations(common_cols, i)) for i in range(1,len(common_cols)+1)]

    return lattice

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
    #return lattice

    for i in range(len(lattice)):
        level = lattice[i]
        new_lattice = lattice
        for tup in level:
            contained = check_col_group_containment(df1,df2,tup)
            if not contained:
                #print('removing', tup)
                new_lattice = remove_tup_lattice(new_lattice, tup)
        lattice = new_lattice

        #return lattice

    non_empty = [l for l in lattice if len(l)>0]
    if non_empty:
        return  non_empty[-1][0]
    return []

# Given a Notebook (nb_name) and directory full of csvs (dir), return all probable joins via containment
def get_all_joins_wf(nb_name, csvdir):
    joins = []
    artifact_dir = csvdir
    artifacts = [os.path.basename(p) for p in glob.glob(artifact_dir+'*.csv')]
    df_dict = {artifact: get_dataframe(csvdir, artifact) for artifact in artifacts}
    combos = itertools.combinations(df_dict.keys(),3)
    for combo in combos:
        #print(combo)
        sizes = {x: len(set(df_dict[x])) for x in combo}
        if max(sizes.values())==min(sizes.values()):
            continue
        join_dest = list(sizes.keys())[list(sizes.values()).index(max(sizes.values()))]
        join_sources = tuple(x for x in combo if x is not join_dest)

        if set(df_dict[join_sources[0]]).union(set(df_dict[join_sources[1]])) == set(df_dict[join_dest]):
            # print ('Column Union Match:', join_dest, join_sources)

            coherent_1 = get_max_coherent_columns(df_dict[join_sources[0]], df_dict[join_dest])
            coherent_2 = get_max_coherent_columns(df_dict[join_sources[1]], df_dict[join_dest])

            # Check if the coherent columns generate the output set
            if set(coherent_1).union(set(coherent_2)) == set(df_dict[join_dest]):
                print('coherent:', (join_dest, join_sources))
                if set(coherent_1).intersection(set(coherent_2)): # Check if the intersection is not null
                    print('intersection: ', (join_dest, join_sources))
                    joins.append((join_dest, join_sources))

    return joins


## Group By Detection

def df_groupby_check(df1,df2):
    combinations = itertools.product(list(df1), list(df2))
    for col1,col2 in combinations:
        if(column_groupby_check(df1[col1], df2[col2])):
            return str(col1), str(col2)
    return False

def get_all_groupbys_wf(nb_name, csvdir):
    artifact_dir = csvdir
    artifacts = [os.path.basename(p) for p in glob.glob(artifact_dir+'*.csv')]
    df_dict = {artifact: get_dataframe(nb_name, artifact) for artifact in artifacts}
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
