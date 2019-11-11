import sys
import os
import pandas as pd
import itertools
# import collections
# import merkle_tree
import numpy as np


from glob import glob
from tqdm import tqdm_notebook

import numpy.testing as npt

from datasketch import MinHash


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
    for d1, d2 in tqdm_notebook(pairs, desc='graph pairs', leave=False):
        score = similarity_metric(dataset[d1], dataset[d2])
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            pass
            #print("WARNING: DROPPING",d1,d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity


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
def compute_jaccard_DF(df1,df2, pk_col_name=None):

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

    # Compute Jaccard Similarity
    intersection = np.sum(np.sum(df3))
    union = df3.size
    return float(intersection) / union


#Assumes corresponding column names and valid indices in both data frames
def compute_jaccard_DF_index(df1,df2):

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA')
    df2 = df2.fillna('jac_tmp_NA')

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
    for d1, d2 in tqdm_notebook(pairs, desc='graph pairs', leave=False):
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