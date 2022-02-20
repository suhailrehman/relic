import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
import timeit
import sys
import string
import numpy as np

from valentine import valentine_match, valentine_metrics
from valentine.algorithms import Coma, Cupid, DistributionBased, JaccardLevenMatcher, SimilarityFlooding

from relic.utils.serialize import build_df_dict_dir

algorithm_map = {
    'coma': Coma(strategy="COMA_OPT"),
    'cupid': Cupid(),
    'db': DistributionBased(),
    'jlm': JaccardLevenMatcher(),
    'sf': SimilarityFlooding(),
}


# SCHEMA PERTURBATION FUNCTIONS ####
_UNIQUE_DICTIONARY = string.ascii_letters+string.digits


# Schema Perturbation Function
def generate_prefix(symbol_dict: str=_UNIQUE_DICTIONARY, size: int=5) -> str:
    return ''.join(np.random.choice(list(symbol_dict), size))


def perturb_string(original_string, beta, prefix_change=False):
    if prefix_change:
        return generate_prefix()+'__'+original_string.split('__')[1]
    else:
        strlen = len(original_string)
        new_string=list(original_string)
        pertrub_idxs = np.random.randint(0, strlen, int(np.ceil(beta*strlen))).tolist()
        for ix in pertrub_idxs:
            new_string[ix] = np.random.choice(list(_UNIQUE_DICTIONARY), 1)[0]
        return "".join(new_string)


def perturb_schema(input_df, alpha=0.5, prefix_change=False, beta=0.2):
    original_column_set = set(input_df.columns)
    rename_map = {}
    cols_to_rename = np.random.choice(list(original_column_set), int(np.ceil(alpha*len(original_column_set))))

    for col_to_rename in cols_to_rename:
        rename_map[col_to_rename] = perturb_string(col_to_rename, beta=beta, prefix_change=prefix_change)

    return input_df.rename(columns=rename_map), rename_map


def get_df_pair_ground_truth(df1, df2, rename_map_df1=None, rename_map_df2=None):
    common_columns = set(df1.columns).intersection(set(df2.columns))
    if rename_map_df1 and rename_map_df2:
        return_list = []
        for col in common_columns:
            x = rename_map_df1[col] if col in rename_map_df1 else col
            y = rename_map_df2[col] if col in rename_map_df2 else col
            return_list.append((x, y))
        return return_list
    else:
        return [(x, x) for x in common_columns]


def evaluate_dir(base_dir, nb_name, matcher, matcher_label, alpha=None, prefix_change=False, beta=None):
    csv_dir = f"{base_dir}/{nb_name}/artifacts/"
    graph_file = f"{base_dir}/{nb_name}/{nb_name}_gt_fixed.pkl"
    gt_graph = nx.read_gpickle(graph_file)
    dfs = build_df_dict_dir(csv_dir)
    renamed_dfs = {}
    rename_col_map = {}
    if alpha and beta:
        for label, df in dfs.items():
            renamed_df, rename_map = perturb_schema(df, alpha=alpha, prefix_change=prefix_change, beta=beta)
            renamed_dfs[label] = renamed_df
            rename_col_map[label] = rename_map

    results = []

    for u, v, e_data in tqdm(list(gt_graph.edges(data=True)), desc=f'Pairwise {matcher_label}', leave=False):
        if alpha and beta:
            ground_truth_col_match = get_df_pair_ground_truth(dfs[u], dfs[v], rename_map_df1=rename_col_map[u],
                                                              rename_map_df2=rename_col_map[v])
            start_time = timeit.default_timer()
            matches = valentine_match(renamed_dfs[u], renamed_dfs[v], matcher)
            end_time = timeit.default_timer()
        else:
            ground_truth_col_match = get_df_pair_ground_truth(dfs[u], dfs[v])
            start_time = timeit.default_timer()
            matches = valentine_match(dfs[u], dfs[v], matcher)
            end_time = timeit.default_timer()

        metrics = valentine_metrics.all_metrics(matches, ground_truth_col_match)
        results.append({'nb_name': nb_name,
                        'matcher': matcher_label,
                        'edge': frozenset((u, v)),
                        **metrics,
                        **e_data,
                        'runtime': end_time - start_time,
                        'alpha': alpha,
                        'prefix_change': prefix_change,
                        'beta': beta})

    return pd.DataFrame(results)


if __name__ == '__main__':
    bd = sys.argv[1]
    nb = sys.argv[2]
    output_file = sys.argv[3]
    m = algorithm_map[sys.argv[4]]
    alpha=None
    prefix_change=None
    beta=None
    if len(sys.argv) > 5:
       alpha = float(sys.argv[5])
       prefix_change = (sys.argv[6] == 'true')
       beta = float(sys.argv[7])
    r_df = evaluate_dir(bd, nb, m, sys.argv[4], alpha, prefix_change, beta)
    r_df.to_csv(output_file)
