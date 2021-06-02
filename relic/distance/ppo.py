import numpy as np
import logging

from relic.distance.set_functions import set_jaccard_similarity
from relic.utils.matching import fuzzy_column_match, set_df_indices

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def _return_zero(ppo):
    result_dict = {
        'jaccard': 0.0,
        'containment': 0.0,
        'overlap': 0,
        'containment_oneside': 0.0
    }

    if ppo == 'all':
        return result_dict
    elif ppo in result_dict:
        return {ppo: result_dict[ppo]}
    else:
        raise ValueError('Unknown distance type requested', ppo)


def compute_all_ppo_labels(d1, d2, df_dict, **kwargs):
    return frozenset([d1,d2]), compute_all_ppo(df_dict[d1], df_dict[d2], **kwargs)


# Assumes corresponding column names are same and PK refers to same column.
def compute_all_ppo(df1, df2, ppo='all', pk_col_name=None, reindex=False,
                    column_match=False, string_cols=True):

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA')
    df2 = df2.fillna('jac_tmp_NA')

    # Cast column names to str in both:
    if string_cols:
        df1.columns = df1.columns.map(str)
        df2.columns = df2.columns.map(str)

    try:
        if reindex:
            df1, df2 = set_df_indices(df1, df2)
        if column_match:
            df1, df2 = fuzzy_column_match(df1, df2)
    except IndexError as e:
        logging.warning(f"Reindex error. Ignoring: {e}")
        pass

    try:
        if pk_col_name:
            df3 = df1.merge(df2, how='outer', on=pk_col_name, suffixes=['_jac_tmp_1', '_jac_tmp_2'])
        else:
            df3 = df1.merge(df2, how='outer', left_index=True, right_index=True, suffixes=['_jac_tmp_1', '_jac_tmp_2'])
    except TypeError as e:
        logger.warning('Cannot merge', df1.head(), df2.head(), e)
        return _return_zero(ppo)

    # Get set of column column names:
    comparison_cols = set(col for col in df3.columns if '_jac_tmp_' in str(col))
    common_cols = set(col.split('_jac_tmp_', 1)[0] for col in comparison_cols)

    if len(common_cols) == 0:
        logger.debug(f'No common cols: {comparison_cols}, {common_cols}, {df3.head()}')
        return _return_zero(ppo)

    # Get set of non-common columns:
    uniq_cols = set(col for col in df3.columns if '_jac_tmp_' not in str(col))
    if pk_col_name:
        uniq_cols.remove(pk_col_name)

    # Check common cols and print True/False
    for col in common_cols:
        left = col + '_jac_tmp_1'
        right = col + '_jac_tmp_2'
        try:
            df3[col] = np.isclose(df3[left], df3[right])
        except TypeError:
            df3[col] = df3[left] == df3[right]

    # Unique columns are already false
    for col in uniq_cols:
        df3[col] = False

    # Drop superfluous columns
    df3 = df3.drop(columns=comparison_cols)
    if pk_col_name:
        df3 = df3.drop(columns=[pk_col_name])

    # Compute Jaccard Similarity
    intersection = np.sum(np.sum(df3))
    union = df3.size
    minsize = min(df1.size, df2.size)

    logger.debug(f'Merged: {df3.head()}')

    result_dict = {
        'jaccard': float(intersection) / union,
        'containment': float(intersection) / minsize,
        'overlap': float(intersection),
        'containment_oneside': float(intersection) / df2.size
    }

    if ppo == 'all':
        return result_dict
    elif ppo in result_dict:
        return {ppo: result_dict[ppo]}
    else:
        raise ValueError("Unknown ppo type requested:", ppo)


# Assumes corresponding column names and valid indices in both data frames
def compute_col_jaccard_df(d1, d2, df_dict):
    df1 = df_dict[d1]
    df2 = df_dict[d2]
    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA').reset_index(drop=True)
    df2 = df2.fillna('jac_tmp_NA').reset_index(drop=True)

    common_cols = set(df1).intersection(set(df2))

    if len(common_cols) == 0:
        return 0.0

    common_cols_jaccard = []

    # Check common cols and print True/False
    for col in common_cols:
        try:
            sim = set_jaccard_similarity(set(df1[col].values), set(df2[col].values))
            common_cols_jaccard.append(sim)
            logging.debug(f'col: {col} jaccard: {sim}')

        except Exception as e:
            print(col)
            print(set_jaccard_similarity(set(df1[col].values), set(df2[col].values)))
            raise e

    logging.debug(f'num/denom: {np.sum(common_cols_jaccard)}, {len(set(df1).union(set(df2)))}')
    return np.sum(common_cols_jaccard) / len(set(df1).union(set(df2)))
