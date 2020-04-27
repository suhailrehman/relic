import networkx as nx

import pandas as pd
import numpy as np
import os

from lineage import similarity, graphs, precomputed_sim

import dataset as ds
import clustering

import nppo

from PIL import Image

import sys
import argparse
import timeit


#BASE_DIR = '/home/suhail/Projects/sample_workflows/million_notebooks/selected/'
BASE_DIR = '/home/suhail/Projects/relic/primitives/python/generator/dataset/'

NB_NAME = 'nb_331056.ipynb'

notebooks = [
    'nb_331056.ipynb',
    'nb_495072.ipynb',
    'nb_315236.ipynb',
    'churn',
    'githubviz',
    'titanic',
    'retail'
]

notebooks = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

def append_result(pr_df, df_dict, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster, time):
    result = graphs.get_precision_recall(g_truth, g_inferred)

    if '0.csv' not in df_dict:
        try:
            root_node = [x for x in nx.topological_sort(g_truth)][0] #TODO: Check more than one root issues
        except nx.exception.NetworkXUnfeasible as e:
            print("ERROR: Cycle in Graph")
            root_node = list(df_dict.keys())[0]
            pass
    else:
        root_node = '0.csv'

    pr_df = pr_df.append({
        'nb_name': nb_name,
        'rows': df_dict[root_node].shape[0],
        'columns': df_dict[root_node].shape[1],
        'artifacts': len(df_dict),
        'index': index,
        'pre_cluster': pre_cluster,
        'numclusters': len(clusters),
        'distance_metric': 'pandas_col',
        'edges_correct': len(result['correct_edges']),
        'edges_missing': len(result['to_add']),
        'edges_to_remove': len(result['to_remove']),
        # 'join_edges': len(inferred_j_edges),
        'precision': result['Precision'],
        'recall': result['Recall'],
        'F1': result['F1'],
        'missing_files': len(missing_files),
        'time': time
    }, ignore_index=True)

    return pr_df


def lineage_inference_baseline(nb_name=NB_NAME, base_dir=BASE_DIR,
                                    pre_cluster='No Precluster',
                                    intra_cell_threshold=0.1,
                                    inter_cell_threshold=0.1,
                                    join_edges=False,
                                    group_edges=False,
                                    index=False,
                                    draw=False
                                    ):
    print('Processing:', nb_name)

    start_time = timeit.default_timer()

    wf_dir = base_dir + nb_name
    artifact_dir = wf_dir+'/artifacts/'


    # Output Directory
    result_dir = wf_dir + '/inferred/'
    os.makedirs(result_dir, exist_ok=True)

    # Output Files
    schema_file = result_dir + 'schema_matching.csv'
    row_file = result_dir + 'row_matching.csv'
    cluster_file = result_dir + 'clusters.csv'

    # Prepare Dataframe for results
    pr_df = pd.DataFrame(columns=['nb_name', 'rows', 'columns', 'artifacts', 'index', 'numclusters',
                                  'distance_metric', 'edges_correct',
                                  'edges_missing', 'edges_to_remove',
                                  'join_edges', 'precision', 'recall', 'F1',
                                  'missing_files', 'time'])

    # Image Array for Animated GIF
    img_frames = []

    # Load Dataset
    dataset = ds.build_df_dict_dir(artifact_dir)

    # Load Ground Truth:
    g_truth = nx.read_gpickle(wf_dir + '/' + nb_name + '_gt_fixed.pkl')

    # Write ground truth image
    if draw:
        graphs.generate_notebook_image(base_dir, nb_name)

    # Compute all-pairs similarity for visualization
    # Start with intra-cluster edges:
    all_pairwise_jaccard = similarity.get_pairwise_similarity(dataset, similarity.compute_jaccard_DF)
    all_pw_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_jaccard)

    # Write out the Pairwise Distances as Adj list
    nx.to_pandas_adjacency(all_pw_jaccard_graph, weight='weight').to_csv(result_dir + 'cell_sim.csv')

    # Check for files in the ground truth that are missing in file list
    missing_files = ds.check_csv_graph(artifact_dir, g_truth)

    # Cluster for visualization
    clusters = clustering.exact_schema_cluster(dataset)
    clustering.write_clusters_to_file(clusters, result_dir + 'clusters_with_filename.csv')
    cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')

    # Baseline max spanning tree
    g_inferred = nx.maximum_spanning_tree(all_pw_jaccard_graph)

    # Draw first graph and get results
    nx.write_edgelist(g_inferred, result_dir + 'infered_mst_baseline_cell.csv', data=True)
    if draw:
        img_frames.append(graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict,
                                                         join_list=None))

    pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster,
                          timeit.default_timer() - start_time)

    image_frames = [Image.open(frame) for frame in img_frames]

    if draw:
        image_frames[0].save(base_dir+nb_name+'/baseline_union_construction.gif',
                             format='GIF', append_images=image_frames[1:],
                             save_all=True,
                             duration=1000,
                             loop=0)

    return pr_df, image_frames


def experiment_1(base_dir, nb_name, clustering):
    pd.set_option('display.max_columns', None)

    result, im_frames = lineage_inference_baseline(base_dir=base_dir, nb_name=nb_name, pre_cluster=clustering)
    print(result[['numclusters', 'edges_correct', 'edges_missing', 'edges_to_remove', 'F1', 'time']])
    result.to_csv(base_dir + nb_name + '/baseline_union_result.csv')

    return result


def setup_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--basedir",
                        help="Base Directory of Datasets to be analysed",
                        type=str, default=BASE_DIR)

    parser.add_argument("--nbname",
                        help="Name of the notebook to be analyzed",
                        type=str, default=notebooks[0])

    parser.add_argument("--clustering",
                        help="Clustering Method",
                        type=str, default='No Preclustering')


    options = parser.parse_args(args)

    return options


def main(args=sys.argv[1:]):

    options = setup_arguments(args)
    experiment_1(options.basedir, options.nbname, options.clustering)



if __name__ == "__main__":
    main()
