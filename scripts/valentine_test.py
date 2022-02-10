import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
import timeit
import sys

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


def get_df_pair_ground_truth(df1, df2):
    common_columns = set(df1.columns).intersection(set(df2.columns))
    return [(x, x) for x in common_columns]


def evaluate_dir(base_dir, nb_name, matcher, matcher_label):
    csv_dir = f"{base_dir}/{nb_name}/artifacts/"
    graph_file = f"{base_dir}/{nb_name}/{nb_name}_gt_fixed.pkl"
    gt_graph = nx.read_gpickle(graph_file)
    dfs = build_df_dict_dir(csv_dir)
    results = []

    for u, v, e_data in tqdm(list(gt_graph.edges(data=True)), desc=f'Pairwise {matcher_label}', leave=False):
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
                        'runtime': end_time - start_time})

    return pd.DataFrame(results)


if __name__ == '__main__':
    bd = sys.argv[1]
    nb = sys.argv[2]
    output_file = sys.argv[3]
    m = algorithm_map[sys.argv[4]]
    r_df = evaluate_dir(bd, nb, m, sys.argv[4])
    r_df.to_csv(output_file)
