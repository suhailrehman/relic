import logging

import networkx as nx
from fuzzywuzzy import fuzz

from relic.distance.set import set_jaccard_similarity


def fuzzy_column_match(df1, df2):
    #print('Performing Fuzzy Match')
    left_side = {col: set(df1[col].values) for col in df1}
    right_side = {col: set(df2[col].values) for col in df2}

    left_mismatch = set(left_side.keys()) - set(right_side.keys())
    right_mismatch = set(right_side.keys()) - set(left_side.keys())

    if left_mismatch and right_mismatch:
        g = nx.DiGraph()
        for l in left_mismatch:
            for r in right_mismatch:
                logging.debug(l,r, fuzz.ratio(l,r))
                g.add_edge(l, r, weight=fuzz.WRatio(l, r))
        try:
            matching = nx.max_weight_matching(g)
            ordered_match = {(l if l in df1 else r): (r if l in df1 else l) for l, r in matching}

            df1 = df1.rename(columns=ordered_match)
        except AssertionError as e:
            logging.warning('Could not perform column fuzzy matching:', e)
            pass
    return df1, df2


def find_best_index_match(df1, index):
    col_sets = {col: set_jaccard_similarity(index, set(df1[col].values)) for col in df1}
    return sorted(col_sets.items(), key=lambda x: x[1], reverse=True)[0]


def set_df_indices(df1, df2, indexing_threshold=0.5):
    index1 = set(df1.index.values)
    index2 = set(df2.index.values)

    #print("Checking Indices")
    if set_jaccard_similarity(index1, index2) < indexing_threshold:
        #print("")
        df2_col, value1 = find_best_index_match(df2,index1)
        df1_col, value2 = find_best_index_match(df1,index2)
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