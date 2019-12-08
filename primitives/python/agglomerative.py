import networkx as nx

import pandas as pd
import os

from lineage import similarity, graphs, precomputed_sim

import dataset as ds
import clustering

import nppo

from PIL import Image



BASE_DIR = '/home/suhail/Projects/sample_workflows/million_notebooks/selected/'
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

def append_result(pr_df, g_truth, g_inferred, nb_name, index, clusters, missing_files):
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

    return pr_df

def lineage_inference_agglomerative(nb_name=NB_NAME, base_dir=BASE_DIR,
                                    pre_cluster='No Precluster',
                                    intra_cell_threshold=0.0001,
                                    inter_cell_threshold=0.0001,
                                    join_edges=False,
                                    group_edges=False,
                                    index=False
                                    ):
    print('Processing:', nb_name)

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

    # Write ground truth image
    graphs.generate_notebook_image(base_dir, nb_name)

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
    cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')

    # Start with intra-cluster edges:
    if(pre_cluster != 'No Precluster'):
        pairwise_jaccard = precomputed_sim.intra_cluster_similarity_pc(dataset, clusters, all_pw_jaccard_graph, threshold=intra_cell_threshold)
    else:
        pairwise_jaccard = all_pairwise_jaccard

    pw_jaccard_graph = graphs.generate_pairwise_graph(pairwise_jaccard)

    g_inferred = graphs.generate_spanning_tree(pw_jaccard_graph)

    # Add vertices to the graph if they don't exist
    for artifact in dataset.keys():
        if artifact not in [n for n in g_inferred.nodes()]:
            #print('Adding artifact to graph', artifact)
            g_inferred.add_node(artifact)

    for edge in g_inferred.edges(data=True):
        print('Adding Intra-Cluster Tree Edge:', edge[0], edge[1], "cell-level score:", edge[2]['weight'])

    # Draw first graph and get results
    img_frames.append(
        graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict, join_list=None))

    pr_df = append_result(pr_df, g_truth, g_inferred, nb_name, index, clusters, missing_files)

    # Now try adding columnar edges to each disconnected cluster subgraph:
    if pre_cluster == 'PC2':
        g_inferred = intra_cluster_add_col_edges(dataset, clusters, g_inferred, all_pw_jaccard_graph, threshold=intra_cell_threshold)
        # Draw first graph and get results
        img_frames.append(
            graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict, join_list=None))
        pr_df = append_result(pr_df, g_truth, g_inferred, nb_name, index, clusters, missing_files)

    print("Finished adding all intra-cluster edges")
    nx.write_edgelist(g_inferred, result_dir + 'infered_mst_cell.csv', data=True)

    # Write out inferred graph
    # nx.write_edgelist(g_inferred,result_dir+'infered_mst_cell.csv',data=True)

    components = [c for c in nx.connected_components(g_inferred)]
    #print('Components:', components)

    steps = 0
    stop = False

    if pre_cluster == 'No Precluster':
        stop = True

    # Clustering Loop Starts here
    while (len(components) > 1 and steps < 20 and not stop):

        steps += 1

        new_graph = clustering.find_components_join_edge(g_inferred, dataset, pw_graph=all_pw_jaccard_graph, threshold=inter_cell_threshold)
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
        pr_df = append_result(pr_df, g_truth, g_inferred, nb_name, index, components, missing_files)

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

        pr_df = append_result(pr_df, g_truth, g_inferred, nb_name, index, clusters, missing_files)

    group_list = None

    if group_edges:
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        groupbys = nppo.get_all_groupbys_dfdict(dataset)
        group_list = [(x[0], x[2]) for x in groupbys]
        pp.pprint(groupbys)
        g_inferred = nppo.add_group_edges(group_list, g_inferred)

        # Check Group Precision/Recall

        result = graphs.get_precision_recall(g_truth, g_inferred)

        cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')
        img_frames.append(
            graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict, join_list=join_list))

        pr_df = append_result(pr_df, g_truth, g_inferred, nb_name, index, clusters, missing_files)

    image_frames = [Image.open(frame) for frame in img_frames]

    image_frames[0].save(BASE_DIR+nb_name+'/relic_construction.gif',
                         format='GIF', append_images=image_frames[1:],
                         save_all=True,
                         duration=1000,
                         loop=0)

    return pr_df, image_frames


# Note: Edges in g_inferred should only be within a cluster
def intra_cluster_add_col_edges(df_dict, clusters, g_inferred, all_pw_jaccard_graph, threshold=0.01):

    for cluster in clusters.values():
        subgraph = g_inferred.subgraph(cluster).copy()
        components = [c for c in nx.connected_components(subgraph)]

        steps = 0
        stop = False

        # Clustering Loop Starts here
        while (len(components) > 1 and steps < 20 and not stop):

            steps += 1

            new_graph = clustering.find_components_join_edge(subgraph, df_dict, pw_graph=all_pw_jaccard_graph, threshold=threshold)
            if not new_graph:
                stop = True
            else:
                subgraph = new_graph

            components = [c for c in nx.connected_components(subgraph)]
            # print('Components:', components)


        for edge in subgraph.edges(data=True):
            g_inferred.add_edge(edge[0], edge[1], weight=edge[2])


    return g_inferred

def main():
    pd.set_option('display.max_columns', None)

    nbs = notebooks[:1]
    for nb_name in nbs:
        result, im_frames = lineage_inference_agglomerative(nb_name=nb_name, pre_cluster='PC2')
        print(result[['numclusters', 'edges_correct', 'edges_missing', 'edges_to_remove', 'F1']])
        result.to_csv(BASE_DIR + nb_name + '/relic_agglomerative_result.csv')



if __name__ == "__main__":
    main()

