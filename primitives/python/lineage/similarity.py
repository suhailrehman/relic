import sys
import os
import pandas as pd
import itertools
# import collections
# import merkle_tree
import numpy as np
import networkx as nx

from glob import glob
from tqdm.autonotebook import tqdm

import numpy.testing as npt

from datasketch import MinHash

from fuzzywuzzy import fuzz


# TODO: clean up project structure
sys.path.append('../merkle/')
sys.path.append('../lsh_forest/')


# Read directory path and glob pattern and retrun a dict of dataframes
# for each file inside
def load_dataset_dir(dirpath, glob_pattern, **kwargs):
    dataset = {}
    for filename in glob(dirpath+glob_pattern):
        dataset[os.path.basename(filename)] = pd.read_csv(filename, **kwargs)
    return dataset


# Compute pairwise similarity metrics of dataset dict using similarity_metric
# returns reverse sorted list of (pair1, pair2, similarity_score) tuples
def get_pairwise_similarity(dataset, similarity_metric, threshold=-1.0):
    pairwise_similarity = []
    pairs = list(itertools.combinations(dataset.keys(), 2))
    for d1, d2 in tqdm(pairs, desc='graph pairs', leave=False):
        score = similarity_metric(dataset[d1], dataset[d2])
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            #pass
            print("WARNING: DROPPING",d1,d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity


def load_pairwise_similarity_from_file(filename):
    adj_list = pd.read_csv(filename, index_col=0)
    g = nx.from_pandas_adjacency(adj_list)
    return g


# Jaccard Functions
# Compute Raw Jaccard Similarity between two dataframes
# SLOWWW
def get_jaccard_coefficient_slow(df1, df2):
    rowsize = max(df1.shape[0], df2.shape[0])
    colsize = max(df1.shape[1], df2.shape[1])

    # print total

    intersection = 0.0
    for i in range(rowsize):
        for j in range(colsize):
            try:
                try:
                    npt.assert_equal(df1.iloc[i][j], df2.iloc[i][j])
                    intersection += 1
                except AssertionError as e:
                    pass
            except IndexError as e:
                pass

    # print intersection

    union = (df1.size + df2.size) - intersection

    return intersection / union


# FASTER but FillNa might pose problems.
def get_jaccard_coefficient(df1, df2):
    minshape = np.minimum(df1.shape, df2.shape)
    iM = np.equal(df1.fillna(np.NINF).values[:minshape[0], :minshape[1]],
                  df2.fillna(np.NINF).values[:minshape[0], :minshape[1]])
    intersection = np.sum(iM)
    union = (df1.size + df2.size) - intersection
    return float(intersection) / union

def get_minhash_coefficient(df1,df2):
    pass


#Assumes corresponding column names are same and PK refers to same column.
def compute_jaccard_DF(df1,df2, pk_col_name=None, reindex=False, column_match=False):

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA')
    df2 = df2.fillna('jac_tmp_NA')

    try:
        if reindex:
            df1, df2 = set_df_indices(df1, df2)

        if column_match:
            df1, df2 = fuzzy_column_match(df1, df2)
    except IndexError as e:
        print("Reindex error. Ignoring.")
        pass

    try:
        if(pk_col_name):
            df3 = df1.merge(df2, how='outer', on=pk_col_name, suffixes=['_jac_tmp_1','_jac_tmp_2'])
        else:
            df3 = df1.merge(df2, how='outer', left_index=True, right_index=True, suffixes=['_jac_tmp_1','_jac_tmp_2'])
    except TypeError as e:
        # print("Can't Merge")
        return 0

    # Get set of column column names:
    comparison_cols = set(col for col in df3.columns if'_jac_tmp_' in str(col))
    common_cols = set(col.split('_jac_tmp_',1)[0] for col in comparison_cols)

    if(len(common_cols) == 0):
        return 0

    # Get set of non-common columns:
    uniq_cols = set(col for col in df3.columns if'_jac_tmp_' not in str(col))
    if(pk_col_name):
        uniq_cols.remove(pk_col_name)

    # Check common cols and print True/False
    for col in common_cols:
        left = col+'_jac_tmp_1'
        right = col+'_jac_tmp_2'
        df3[col] = df3[left] == df3[right]

    # Unique columns are already false
    for col in uniq_cols:
        df3[col] = False

    #Drop superflous columns
    df3 = df3.drop(columns=comparison_cols)
    if(pk_col_name):
        df3 = df3.drop(columns=[pk_col_name])

    # Compute Jaccard Similarity
    intersection = np.sum(np.sum(df3))
    union = df3.size
    return float(intersection) / union


#Assumes corresponding column names and valid indices in both data frames
def compute_jaccard_DF_index(df1,df2, reindex=True):

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA')
    df2 = df2.fillna('jac_tmp_NA')

    if reindex:
        df1, df2 = set_df_indices(df1, df2)

    df3 = df1.merge(df2, how='outer', left_index=True, right_index=True, suffixes=['_jac_tmp_1','_jac_tmp_2'])

    # Get set of column column names:
    comparison_cols = set(col for col in df3.columns if'_jac_tmp_' in str(col))
    common_cols = set(col.split('_jac_tmp_',1)[0] for col in comparison_cols)

    if(len(common_cols) == 0):
        return 0

    # Get set of non-common columns:
    uniq_cols = set(col for col in df3.columns if'_jac_tmp_' not in str(col))

    # Check common cols and print True/False
    for col in common_cols:
        try:
            left = col+'_jac_tmp_1'
            right = col+'_jac_tmp_2'
            df3[col] = df3[left] == df3[right]
        except Exception as e:
            print(col, left, right)
            print(df3[left] == df3[right])
            raise e

    # Unique columns are already false
    for col in uniq_cols:
        df3[col] = False

    #Drop superflous columns
    df3 = df3.drop(columns=comparison_cols)

    # Compute Jaccard Similarity
    intersection = np.sum(np.sum(df3))
    union = df3.size
    if(union == 0):
        return 0.0

    del(df3)

    return float(intersection) / union

# Assumes corresponding column names and valid indices in both data frames
def compute_jaccard_DF_reindex(df1,df2):

    # Empty DF check

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA').reset_index(drop=True)
    df2 = df2.fillna('jac_tmp_NA').reset_index(drop=True)

    df3 = df1.merge(df2, how='outer', left_index=True, right_index=True, suffixes=['_jac_tmp_1','_jac_tmp_2'])

    # Get set of column column names:
    comparison_cols = set(col for col in df3.columns if'_jac_tmp_' in str(col))
    common_cols = set(col.split('_jac_tmp_',1)[0] for col in comparison_cols)

    if(len(common_cols) == 0):
        return 0.0

    # Get set of non-common columns:
    uniq_cols = set(col for col in df3.columns if'_jac_tmp_' not in str(col))

    # Check common cols and print True/False
    for col in common_cols:
        try:
            left = col+'_jac_tmp_1'
            right = col+'_jac_tmp_2'
            df3[col] = df3[left] == df3[right]
        except Exception as e:
            print(col, left, right)
            print(df3[left] == df3[right])
            raise e

    # Unique columns are already false
    for col in uniq_cols:
        df3[col] = False

    #Drop superflous columns
    df3 = df3.drop(columns=comparison_cols)

    # Compute Jaccard Similarity
    intersection = np.sum(np.sum(df3))
    union = df3.size
    if(union == 0):
        return 0.0

    del(df3)

    return float(intersection) / union



#Assumes corresponding column names are same and PK refers to same column.
def compute_DF_overlap(df1,df2, pk_col_name=None):

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA')
    df2 = df2.fillna('jac_tmp_NA')

    try:
        if(pk_col_name):
            df3 = df1.merge(df2, how='outer', on=pk_col_name, suffixes=['_jac_tmp_1','_jac_tmp_2'])
        else:
            df3 = df1.merge(df2, how='outer', left_index=True, right_index=True, suffixes=['_jac_tmp_1','_jac_tmp_2'])
    except TypeError as e:
        # print("Can't Merge")
        return 0

    # Get set of column column names:
    comparison_cols = set(col for col in df3.columns if'_jac_tmp_' in str(col))
    common_cols = set(col.split('_jac_tmp_',1)[0] for col in comparison_cols)

    if(len(common_cols) == 0):
        return 0

    # Get set of non-common columns:
    uniq_cols = set(col for col in df3.columns if'_jac_tmp_' not in str(col))
    if(pk_col_name):
        uniq_cols.remove(pk_col_name)

    # Check common cols and print True/False
    for col in common_cols:
        left = col+'_jac_tmp_1'
        right = col+'_jac_tmp_2'
        df3[col] = df3[left] == df3[right]

    # Unique columns are already false
    for col in uniq_cols:
        df3[col] = False

    #Drop superflous columns
    df3 = df3.drop(columns=comparison_cols)
    if(pk_col_name):
        df3 = df3.drop(columns=[pk_col_name])

    # Return Intersection
    intersection = np.sum(np.sum(df3))
    return intersection



def get_pairs_similarity(dataset, cluster_set1, cluster_set2, similarity_metric=compute_jaccard_DF, threshold=-1.0):
    pairwise_similarity = []
    pairs = list(itertools.product(cluster_set1, cluster_set2))
    for d1, d2 in pairs:
        if d1 == d2:
            continue
        score = similarity_metric(dataset[d1], dataset[d2])
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            pass
            #print("WARNING: DROPPING",d1,d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity


def intra_cluster_similarity(df_dict, clusters, threshold=0.25):
    pairwise_jaccard = []
    for cluster in clusters.values():
        batch = {k: df_dict[k] for k in cluster}
        pw_batch = get_pairwise_similarity(batch, compute_jaccard_DF, threshold=threshold)
        pairwise_jaccard.extend(pw_batch)
    return pairwise_jaccard



# Duplicate function
def set_jaccard_distance(set1, set2):
    return 1 - set_jaccard_similarity(set1,set2)


# Duplicate function
def set_jaccard_similarity(set1, set2):
    intersect = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersect) / len(union)

def set_max_containment(set1,set2):
    intersect = len(set1.intersection(set2))
    return max(intersect/len(set1), intersect/len(set2))


# Assumes corresponding column names and valid indices in both data frames
def compute_col_jaccard_DF(df1,df2):

    # Empty DF check

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA').reset_index(drop=True)
    df2 = df2.fillna('jac_tmp_NA').reset_index(drop=True)

    common_cols = set(df1).intersection(set(df2))

    if(len(common_cols) == 0):
        return 0.0


    common_cols_jaccard = []

    # Check common cols and print True/False
    for col in common_cols:
        try:
            common_cols_jaccard.append(set_jaccard_similarity(set(df1[col].values), set(df2[col].values)))
        except Exception as e:
            print(col)
            print(set_jaccard_similarity(set(df1[col].values), set(df2[col].values)))
            raise e

    # return np.average(common_cols_jaccard)
    return np.sum(common_cols_jaccard) / len(set(df1).union(set(df2)))


def compute_col_jaccard_DF_union(df1,df2):

    # Empty DF check

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA').reset_index(drop=True)
    df2 = df2.fillna('jac_tmp_NA').reset_index(drop=True)

    common_cols = set(df1).intersection(set(df2))

    if(len(common_cols) == 0):
        return 0.0


    common_cols_jaccard = []

    # Check common cols and print True/False
    for col in common_cols:
        try:
            common_cols_jaccard.append(set_jaccard_similarity(set(df1[col].values), set(df2[col].values)))
        except Exception as e:
            print(col)
            print(set_jaccard_similarity(set(df1[col].values), set(df2[col].values)))
            raise e

    return np.sum(common_cols_jaccard) / len(set(df1).union(set(df2)))



def find_best_index_match(df1, df2, index):
    col_sets = {col: set_jaccard_similarity(index, set(df2[col].values)) for col in df2}
    return sorted(col_sets.items(), key=lambda x: x[1], reverse=True)[0]

def set_df_indices(df1, df2, indexing_threshold=0.5):
    index1 = set(df1.index.values)
    index2 = set(df2.index.values)

    #print("Checking Indices")
    if set_jaccard_similarity(index1, index2) < indexing_threshold:
        #print("")
        df2_col, value1 = find_best_index_match(df1,df2,index1)
        df1_col, value2 = find_best_index_match(df2,df1,index2)

        if value1 < indexing_threshold and value2 < indexing_threshold:
            #TODO: Check if index is autonumbered and do something else
            #print("Resetting both indices")
            df1 = df1.reset_index()
            df2 = df2.reset_index()
        elif value1 > value2:
            #print("Resetting DF2 index to ", df2_col, value1)
            df2 = df2.set_index(df2_col, inplace=False)
        else:
            #print("Resetting DF1 index to ", df1_col, value2)
            df1 = df1.set_index(df1_col, inplace=False)

    return df1, df2


def fuzzy_column_match(df1, df2):
    #print('Performing Fuzzy Match')
    left_side = {col: set(df1[col].values) for col in df1}
    right_side = {col: set(df2[col].values) for col in df2}

    # print(left_side)
    # print(right_side)
    left_mismatch = set(left_side.keys()) - set(right_side.keys())
    right_mismatch = set(right_side.keys()) - set(left_side.keys())

    # print(left_mismatch)
    # print(right_mismatch)

    if left_mismatch and right_mismatch:
        g = nx.DiGraph()
        for l in left_mismatch:
            for r in right_mismatch:
                # print(l,r, fuzz.ratio(l,r))
                g.add_edge(l, r, weight=fuzz.WRatio(l, r))

        # print(g.edges(data=True))

        try:
            matching = nx.max_weight_matching(g)

            ordered_match = {(l if l in df1 else r): (r if l in df1 else l) for l, r in matching}

            df1 = df1.rename(columns=ordered_match)
        except AssertionError as e:
            #print('Could not match columns')
            pass

    return df1, df2


## VALSET - Flat value similiarity

def get_df_valset(df):
    valset = set()
    for col in df:
        valset = valset.union(set(df[col].values))
    return valset


def compute_valset_similarity(df1, df2):
    return set_jaccard_similarity(get_df_valset(df1), get_df_valset(df2))


def compute_valset_similarity_dict(df1, df2, valset_dict):
    return set_jaccard_similarity(valset_dict[df1], valset_dict[df2])


def generate_valset_dict(df_dict):
    return {name: get_df_valset(df) for name, df in df_dict.items()}


def generate_indexed_valset_dict(df_dict):
    return {name: get_indexed_valset(df) for name, df in df_dict.items()}

def generate_indexed_colvalset_dict(df_dict):
    return {name: get_indexed_colvalset(df) for name, df in df_dict.items()}

def generate_indexed_cellvalset_dict(df_dict):
    return {name: get_indexed_cellvalset(df) for name, df in df_dict.items()}



def get_pairwise_similarity_cellvalset(dataset, threshold=-1.0):
    pairwise_similarity = []
    pairs = list(itertools.combinations(dataset.keys(), 2))

    valset_dict = generate_indexed_cellvalset_dict(dataset)

    for d1, d2 in tqdm(pairs, desc='graph pairs', leave=False):
        score = compute_valset_similarity_dict(d1, d2, valset_dict)
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            # pass
            print("WARNING: DROPPING", d1, d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity


def get_pairwise_similarity_colvalset(dataset, threshold=-1.0):
    pairwise_similarity = []
    pairs = list(itertools.combinations(dataset.keys(), 2))

    valset_dict = generate_indexed_colvalset_dict(dataset)

    for d1, d2 in tqdm(pairs, desc='graph pairs', leave=False):
        score = compute_valset_similarity_dict(d1, d2, valset_dict)
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            # pass
            print("WARNING: DROPPING", d1, d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity


def get_pairwise_similarity_valset(dataset, threshold=-1.0, indexed=False):
    pairwise_similarity = []
    pairs = list(itertools.combinations(dataset.keys(), 2))

    if indexed:
        valset_dict = generate_indexed_valset_dict(dataset)
    else:
        valset_dict = generate_valset_dict(dataset)

    for d1, d2 in tqdm(pairs, desc='graph pairs', leave=False):
        score = compute_valset_similarity_dict(d1, d2, valset_dict)
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            # pass
            print("WARNING: DROPPING", d1, d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity



def generate_frozen_set(col):
    col = col.fillna('jac_tmp_NA')
    return set((i,v) for i, v in col.iteritems())

def get_indexed_valset(df):
    all_values = set()
    for col in df:
        all_values = all_values.union(generate_frozen_set(df[col]))
    return all_values

def get_indexed_colvalset(df):
    all_values = set()
    df = df.fillna('jac_tmp_NA')
    for col in df:
        all_values = all_values.union(set((col, val) for val in set(df[col].values)))
    return all_values


def get_indexed_cellvalset(df):
    all_values = set()
    df = df.fillna('jac_tmp_NA')
    for col in df:
        all_values = all_values.union(set((i, col, val) for i, val in set(df[col].iteritems())))
    return all_values


def compute_indexed_valset_similarity(df1, df2):
    return set_jaccard_similarity(get_indexed_valset(df1), get_indexed_valset(df2))


def compute_indexed_colvalset_similarity(df1, df2):
    return set_jaccard_similarity(get_indexed_colvalset(df1), get_indexed_colvalset(df2))


def compute_indexed_cellvalset_similarity(df1, df2):
    return set_jaccard_similarity(get_indexed_cellvalset(df1), get_indexed_cellvalset(df2))
