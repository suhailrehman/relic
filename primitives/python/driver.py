import networkx as nx
import math
import matplotlib.pyplot as plt
from networkx.algorithms import tree
import pandas as pd
import glob
import os

from lineage import similarity, graphs

import dataset as ds
import clustering

import nppo

BASE_DIR = '/media/suhail/Data/experiments/reexec/res/'
NB_NAME = 'githubviz'


def lineage_inference(nb_name=NB_NAME, base_dir=BASE_DIR,
                      pre_cluster=False,
                      index=True, threshold=0.0001,
                      join_edges=False,
                      group_edges=False,
                      ):

    wf_dir = base_dir+nb_name

    if index:
        artifact_dir = wf_dir+'/artifacts/'
    else:
        artifact_dir = wf_dir+'/artifacts_1/'

    #Output Directory
    result_dir = wf_dir+'/inferred/'
    os.makedirs(result_dir, exist_ok=True)

    # Output Files
    schema_file = result_dir+'schema_matching.csv'
    row_file = result_dir+'row_matching.csv'
    cluster_file = result_dir+'clusters.csv'

    # Load Dataset
    dataset = ds.build_df_dict_dir(artifact_dir)

    # Cluster for visualization
    clusters = clustering.exact_schema_cluster(dataset)
    clustering.write_clusters_to_file(clusters, result_dir+'clusters_with_filename.csv')

    # Run the inference
    if pre_cluster:
        pairwise_jaccard = []
        for cluster in clusters.values():
            batch = {k: dataset[k] for k in cluster}
            pw_batch = similarity.get_pairwise_similarity(batch, similarity.compute_jaccard_DF, threshold=threshold)
            pairwise_jaccard.extend(pw_batch)
    else:
        pairwise_jaccard = similarity.get_pairwise_similarity(dataset, similarity.compute_jaccard_DF, threshold=threshold)


    # Write out the Pairwise Distances as Adj list
    pw_jaccard_graph = graphs.generate_pairwise_graph(pairwise_jaccard)
    nx.to_pandas_adjacency(pw_jaccard_graph,weight='weight').to_csv(
                                                result_dir+'cell_sim.csv')

    g_inferred = graphs.generate_spanning_tree(pw_jaccard_graph)
    nx.write_edgelist(g_inferred,result_dir+'infered_mst_cell.csv',data=True)


    # Load Ground Truth:
    g_truth = nx.read_gpickle(wf_dir+'/'+nb_name+'_gt_fixed.pkl')


    # Write ground truth image
    graphs.generate_notebook_image(base_dir, nb_name)

    # Check for files in the ground truth that are missing in file list
    missing_files = ds.check_csv_graph(artifact_dir, g_truth)


    inferred_j_edges = []
    join_list = None
    cluster_dict = None

    if join_edges:
        print('Writing Cluster File')

        print("Adding Join Edges")
        join_list = nppo.find_all_joins_df_dict(dataset)
        print(len(join_list), "Joins Detected")
        g_inferred = nppo.add_join_edges(join_list, g_inferred)


        for join in join_list:
            inferred_j_edges.append((join[0], join[2]))
            inferred_j_edges.append((join[1], join[2]))


        nppo.write_join_candidates(join_list, result_dir+'join_candidates.csv')

        g_truth_j_edges = [(u,v) for u,v,d in g_truth.edges(data=True) \
                           if g_truth[u][v]['operation'] == 'merge' ]

        #Check Join Precision/Recall
        #print(get_join_precision_recall(g_truth_j_edges, inferred_j_edges))



    result = graphs.get_precision_recall(g_truth,g_inferred)


    # Draw inferred graph image:
    cluster_dict = clustering.get_graph_clusters(result_dir+'clusters_with_filename.csv')
    graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict, join_list=join_list)

    pr_df = pd.DataFrame(columns = ['nb_name', 'index', 'preclustering',  'distance_metric', 'edges_correct', 'edges_missing', 'edges_to_remove',
    'join_edges',
                                        'precision', 'recall', 'F1',
                                        'missing_files' ])


    pr_df = pr_df.append({
            'nb_name': nb_name,
            'index': index,
            'preclustering': pre_cluster,
            'distance_metric': 'pandas_cell',
            'edges_correct': len(result['correct_edges']),
            'edges_missing': len(result['to_add']),
            'edges_to_remove': len(result['to_remove']),
            'join_edges': len(inferred_j_edges),
            'precision': result['Precision'],
            'recall': result['Recall'],
            'F1': result['F1'],
            'missing_files': len(missing_files)
        }, ignore_index=True)

    return pr_df


print(lineage_inference(join_edges=True))
