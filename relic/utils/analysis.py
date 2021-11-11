import pandas as pd
import networkx as nx
import os
import glob

import numpy as np
import pickle
import logging
from tqdm.auto import tqdm
import seaborn as sns
import sys

import logging

from relic.graphs.graphs import get_precision_recall  # draw_graph

base_dir = '/mnt/roscoe/data/relic/real-world/'

# For each workflow, load the non-sampled scores (cell_sim.csv )
# and compare to each sampling technique for every ground truth edge
unsampled_result_file ='infered_mst_pc2cell+containment+group+join+pivot.csv'
sample_result_file='relic_inferred_graph.csv'


def result_dict_to_df(result_dict, nb_name='nb', result_type='relic', sampling=1.0, index=False):
    result_dict.update({'nb_name': nb_name,
                        'result_type': result_type,
                        'sampling' : sampling,
                        'index': index})
    return pd.DataFrame({k: [v] for k, v in result_dict.items()})


def graph_to_df(graph, nb_name='nb', score_type='jaccard', sampling=1.0, index=False):
    all_rows = []
    columns = ['nb_name', 'df1', 'df2', score_type, 'sampling', 'index']
    for edge in nx.to_edgelist(graph):
        all_rows.append((nb_name, edge[0], edge[1], edge[2]['weight'], sampling, index))
    return pd.DataFrame(all_rows, columns=columns)


def score_dict_to_df(score_dict, nb_name='nb', score_type='jaccard', sampling=1.0, index=False):
    columns_dict = ['nb_name', 'edge', score_type, 'sampling', 'index']
    all_rows = []
    for edge, score in score_dict.items():
        all_rows.append((nb_name, edge, score, sampling, index))
    return pd.DataFrame(all_rows, columns=columns_dict)


def join_score_dict_to_df(score_dict, nb_name='nb', score_type='join', sampling=1.0, index=False):
    columns_dict = ['nb_name', 'src', 'dst', score_type, 'sampling', 'index']
    all_rows = []
    for fz_edge, edge in score_dict.items():
        (src1, src2), dst, score = edge
        all_rows.append((nb_name, src1, dst, score, sampling, index))
        all_rows.append((nb_name, src2, dst, score, sampling, index))
    return pd.DataFrame(all_rows, columns=columns_dict)


def is_gt_edge(row, gt_graphs):
    u,v = row.edge
    gt = gt_graphs[row['nb_name']].to_undirected()
    if gt.has_edge(u,v):
        return gt[u][v]['operation']
    else:
        return None


def is_gt_edge_join(row, gt_graphs):
    u,v = row.src, row.dst
    gt = gt_graphs[row['nb_name']]#.to_undirected()
    if gt.has_edge(u,v):
        return gt[u][v]['operation']
    else:
        return None


def convert_join_sample_df(join_score_df):
    join_rows = []
    for ix, row in join_score_df.iterrows():
        join_rows.append([row['df1'], row['df2'], row['join']])
    return pd.DataFrame(join_rows, columns=['src', 'dst', 'join'])


def process_directory(base_dir, base_score_file="cell_sim.pkl", score_type='cell', sample_score_file="ppo.csv"):
    workflow_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    all_results = []
    all_gt_graphs = {}
    edges_df = pd.DataFrame()
    relic_result_df = pd.DataFrame()
    # columns = ['nb_name', 'df1', 'df2', score_type, 'sampling', 'index']
    # columns = ['nb_name', 'src', 'dst', score_type, 'sampling', 'index']
    columns = ['nb_name', 'edge', score_type, 'sampling', 'index']

    #print(workflow_ids)
    for i in tqdm(workflow_ids):
        #print(i)
        try:
            gt_file = f"{base_dir}/{i}/{i}_gt_fixed.pkl"
            gt_graph = nx.read_gpickle(gt_file)
            all_gt_graphs[i] = gt_graph
            all_results.append(gt_graph)

            # Load PPO for regular (100%) sample
            inferred_dir = f"{base_dir}/{i}/inferred/"
            cell_sim_g = pickle.load(open(f"{inferred_dir}/{base_score_file}", 'rb'))
            if isinstance(cell_sim_g, dict):
                if score_type == 'join':
                    edges_df = edges_df.append(join_score_dict_to_df(cell_sim_g, nb_name=i, score_type=score_type,
                                                                     sampling=1.0, index='False'))
                    edges_df = edges_df.append(join_score_dict_to_df(cell_sim_g, nb_name=i, score_type=score_type,
                                                                     sampling=1.0, index='True'))
                else:
                    edges_df = edges_df.append(score_dict_to_df(cell_sim_g, nb_name=i, score_type=score_type,
                                                                sampling=1.0, index='False'))
                    edges_df = edges_df.append(score_dict_to_df(cell_sim_g, nb_name=i,
                                                                score_type=score_type, sampling=1.0, index='True'))
            else:
                edges_df = edges_df.append(graph_to_df(cell_sim_g, nb_name=i, score_type=score_type,
                                                       sampling=1.0, index='False'))
                edges_df = edges_df.append(graph_to_df(cell_sim_g, nb_name=i, score_type=score_type,
                                                       sampling=1.0, index='True'))

            # Load Relic Result graph and compute P/R/F1
            relic_graph_file = f"{inferred_dir}/{unsampled_result_file}"
            relic_graph_result = nx.read_edgelist(relic_graph_file)
            result_dict = get_precision_recall(gt_graph, relic_graph_result)
            relic_result_df = relic_result_df.append(result_dict_to_df(result_dict, nb_name=i, index='False'),
                                                     ignore_index=True)
            relic_result_df = relic_result_df.append(result_dict_to_df(result_dict, nb_name=i, index='True'),
                                                     ignore_index=True)

            # Load dfs for each sample type
            for sample_dir in glob.glob(inferred_dir+'/sample_*_*/'):
                sample_dir_name = os.path.basename(sample_dir[:-1])
                _, index, frac = sample_dir_name.split('_')
                try:
                    cell_sim_df = pd.read_csv(f"{sample_dir}/{sample_score_file}", index_col=0)
                    join_df = convert_join_sample_df(cell_sim_df)
                    join_df['index'] = index
                    join_df['sampling'] = float(frac)
                    join_df['nb_name'] = i
                    edges_df = edges_df.append(join_df[columns], ignore_index=True)


                except Exception as e:
                    print(e)
                    raise

        except FileNotFoundError as e:
            logging.warning(f'Could not load File: {e}')
            pass



    # Frozenset edges for easier computation
    edges_df['edge'] = edges_df.apply(lambda x: frozenset([x.src, x.dst]), axis=1)
    # Annotate GT information
    #edges_df['gt_operation'] = edges_df.apply(lambda x: is_gt_edge(x, all_gt_graphs), axis=1)

    return edges_df, all_gt_graphs, relic_result_df


def pivot_and_normalize_edges_df(edges_df, score_col='jaccard', indexed=True, normalize_base_col=(1.0)):
    gt_edges_df = edges_df.loc[edges_df.gt_operation.notna()]
    gt_edges_df = gt_edges_df.loc[gt_edges_df['index'] == str(indexed)]
    results = gt_edges_df.pivot(index=['edge', 'nb_name', 'gt_operation'], columns=['sampling'],
                                values=score_col).dropna()

    # Normalize based on the column labelled by 'normalize_base_col', if present
    if normalize_base_col:
        for col in results.columns:
            base_col = normalize_base_col
            if col != base_col:
                results[col] = results[base_col] - results[col]
            else:
                results[col] = 0
    return results.sort_index(axis=1)


def aggregate_and_display_heatmap(pivoted_normalized_results):
    ppo_heatmap = pivoted_normalized_results.groupby('gt_operation').mean().abs()
    sns.heatmap(ppo_heatmap)
    return ppo_heatmap


def is_join_op(e_data):
    return e_data['operation'] == 'join' or e_data['operation'] == 'merge'


def main(args=sys.argv[1:]):
    synthetic_dir = '/tank/local/suhail/data/relic/relic_datasets_vldb_2021/dataset_flat_exact_sample_timing/'
    logging.info(f'Starting Join Analysis on directory: {synthetic_dir}')
    j_edges_df, gt_graphs, j_relic_result_df = process_directory(synthetic_dir,
                                                                 base_score_file="triple_dict.pkl",
                                                                 score_type='join',
                                                                 sample_score_file="join.csv")
    os.makedirs('/tank/local/suhail/data/relic/relic_datasets_vldb_2021/analysis/dataset_flat_exact_sample_timing/',
                exist_ok=True)
    j_edges_df.to_csv('/tank/local/suhail/data/relic/relic_datasets_vldb_2021/analysis/dataset_flat_exact_sample_timing/joins.csv')
    #pivoted_edges = pivot_and_normalize_edges_df(edges_df, score_col='jaccard', indexed=False, normalize_base_col=(1.0))
    #aggregate_and_display_heatmap(pivoted_edges)


if __name__ == "__main__":
    main()
