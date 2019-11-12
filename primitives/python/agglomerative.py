import networkx as nx

import pandas as pd
import os

from lineage import similarity, graphs, precomputed_sim

import dataset as ds
import clustering

import nppo


BASE_DIR = '/home/suhail/Projects/sample_workflows/million_notebooks/selected/'
NB_NAME = 'nb_331056.ipynb'

def lineage_inference_agglomerative(nb_name=NB_NAME, base_dir=BASE_DIR,
                                    pre_cluster=False,
                                    index=True, threshold=0.0001,
                                    join_edges=False,
                                    group_edges=False,
                                    ):
    wf_dir = base_dir + nb_name

    if index:
        artifact_dir = wf_dir + '/artifacts/'
    else:
        artifact_dir = wf_dir + '/artifacts_1/'

    # Output Directory
    result_dir = wf_dir + '/inferred/'
    os.makedirs(result_dir, exist_ok=True)

    # Output Files
    schema_file = result_dir + 'schema_matching.csv'
    row_file = result_dir + 'row_matching.csv'
    cluster_file = result_dir + 'clusters.csv'

    # Prepare Dataframe for results
    pr_df = pd.DataFrame(columns=['nb_name', 'index', 'numclusters',
                                  'distance_metric', 'edges_correct',
                                  'edges_missing', 'edges_to_remove',
                                  'join_edges', 'precision', 'recall', 'F1',
                                  'missing_files'])

    # Image Array for Animated GIF
    img_frames = []

    # Load Dataset
    dataset = ds.build_df_dict_dir(artifact_dir)

    # Load Ground Truth:
    g_truth = nx.read_gpickle(wf_dir + '/' + nb_name + '_gt_fixed.pkl')

    # Compute all-pairs similarity for visualization
    # Start with intra-cluster edges:
    all_pairwise_jaccard = similarity.get_pairwise_similarity(dataset, similarity.compute_jaccard_DF)
    all_pw_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_jaccard)

    # Write out the Pairwise Distances as Adj list
    nx.to_pandas_adjacency(all_pw_jaccard_graph, weight='weight').to_csv(
        result_dir + 'cell_sim.csv')

    # Check for files in the ground truth that are missing in file list
    missing_files = ds.check_csv_graph(artifact_dir, g_truth)

    # Cluster for visualization
    clusters = clustering.exact_schema_cluster(dataset)
    clustering.write_clusters_to_file(clusters, result_dir + 'clusters_with_filename.csv')

    # Start with intra-cluster edges:
    pairwise_jaccard = precomputed_sim.intra_cluster_similarity_pc(dataset, clusters, all_pw_jaccard_graph)

    pw_jaccard_graph = graphs.generate_pairwise_graph(pairwise_jaccard)

    g_inferred = graphs.generate_spanning_tree(pw_jaccard_graph)

    # Add vertices to the graph if they don't exist
    for artifact in dataset.keys():
        if artifact not in [n for n in g_inferred.nodes()]:
            #print('Adding artifact to graph', artifact)
            g_inferred.add_node(artifact)

    nx.write_edgelist(g_inferred, result_dir + 'infered_mst_cell.csv', data=True)

    # Draw first graph and get results
    cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')
    img_frames.append(
        graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict, join_list=None))

    result = graphs.get_precision_recall(g_truth, g_inferred)

    pr_df = pr_df.append({
        'nb_name': nb_name,
        'index': index,
        'numclusters': len(clusters),
        'distance_metric': 'pandas_cell',
        'edges_correct': len(result['correct_edges']),
        'edges_missing': len(result['to_add']),
        'edges_to_remove': len(result['to_remove']),
        # 'join_edges': len(inferred_j_edges),
        'precision': result['Precision'],
        'recall': result['Recall'],
        'F1': result['F1'],
        'missing_files': len(missing_files)
    }, ignore_index=True)

    # Write out inferred graph
    # nx.write_edgelist(g_inferred,result_dir+'infered_mst_cell.csv',data=True)

    components = [c for c in nx.connected_components(g_inferred)]
    #print('Components:', components)

    steps = 0
    stop = False

    # Clustering Loop Starts here
    while (len(components) > 1 and steps < 20 and not stop):

        steps += 1

        new_graph = clustering.find_components_join_edge(g_inferred, dataset, pw_graph=all_pw_jaccard_graph)
        if not new_graph:
            stop = True
        else:
            g_inferred = new_graph

        components = [c for c in nx.connected_components(g_inferred)]
        #print('Components:', components)

        nx.write_edgelist(g_inferred, result_dir + 'infered_mst_cell.csv', data=True)

        # Draw inferred graph image:
        if (len(clusters) > 1):
            clustering.write_clusters_to_file(clusters, result_dir + 'clusters_with_filename.csv')
            cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')
        else:
            cluster_dict = None
        img_frames.append(
            graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict, join_list=None))

        # Compute PR after merge

        result = graphs.get_precision_recall(g_truth, g_inferred)

        pr_df = pr_df.append({
            'nb_name': nb_name,
            'index': index,
            'numclusters': len(components),
            'distance_metric': 'pandas_cell',
            'edges_correct': len(result['correct_edges']),
            'edges_missing': len(result['to_add']),
            'edges_to_remove': len(result['to_remove']),
            # 'join_edges': len(inferred_j_edges),
            'precision': result['Precision'],
            'recall': result['Recall'],
            'F1': result['F1'],
            'missing_files': len(missing_files)
        }, ignore_index=True)

    # Test for NPPOs:

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

        nppo.write_join_candidates(join_list, result_dir + 'join_candidates.csv')

        g_truth_j_edges = [(u, v) for u, v, d in g_truth.edges(data=True) \
                           if g_truth[u][v]['operation'] == 'merge']

        # Check Join Precision/Recall
        # print(get_join_precision_recall(g_truth_j_edges, inferred_j_edges))

        result = graphs.get_precision_recall(g_truth, g_inferred)

        cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')
        img_frames.append(
            graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict, join_list=join_list))

        pr_df = pr_df.append({
            'nb_name': nb_name,
            'index': index,
            'numclusters': len(clusters),
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

notebooks = [
    'nb_331056.ipynb',
    'nb_495072.ipynb',
    'nb_315236.ipynb',
    'churn',
    'githubviz',
    'titanic',
    'retail'
]

pd.set_option('display.max_columns', None)

for nb_name in notebooks:
    print('Processing:', nb_name)
    result = lineage_inference_agglomerative(nb_name=nb_name, join_edges=False, group_edges=False)
    print(result[['numclusters','edges_correct', 'edges_missing', 'edges_to_remove', 'F1']])
    result.to_csv(BASE_DIR+nb_name+'/relic_agglomerative_result.csv')
