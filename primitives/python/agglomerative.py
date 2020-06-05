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
import itertools


#BASE_DIR = '/home/suhail/Projects/sample_workflows/million_notebooks/selected/'
#BASE_DIR = '/home/suhail/Projects/relic/primitives/python/generator/dataset/'

BASE_DIR="/home/suhail/Projects/sample_workflows/million_notebooks/selected/"


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

def append_result(pr_df, df_dict, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster, time, metric='pandas_cell'):
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
        'distance_metric': metric,
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

def lineage_inference_agglomerative(nb_name=NB_NAME, base_dir=BASE_DIR,
                                    pre_cluster='No Precluster',
                                    cell_threshold=0.10,
                                    col_threshold=0.10,
                                    join_edges=True,
                                    group_edges=True,
                                    transform_edges=True,
                                    pivot_edges=True,
                                    index=False,
                                    draw=False,
                                    recompute=False,
                                    metric='cell',
                                    flip_sim=False
                                    ):


    print('Processing: ', nb_name, ' using metric: '+metric)

    edge_num = 0

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
    if os.path.exists(result_dir+'cell_sim.pkl') and recompute:
        all_pw_jaccard_graph = nx.read_gpickle(result_dir+'cell_sim.pkl')
    else:
        print('Computing pairwise cell similarity')
        all_pairwise_jaccard = similarity.get_pairwise_similarity(dataset, similarity.compute_jaccard_DF)
        all_pw_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_jaccard)
        # Write out the Pairwise Distances as pickled graph
        nx.to_pandas_adjacency(all_pw_jaccard_graph, weight='weight').to_csv(
            result_dir + 'cell_sim.csv')
        nx.write_gpickle(all_pw_jaccard_graph, result_dir + 'cell_sim.pkl')


    if os.path.exists(result_dir+'col_sim.pkl') and recompute:
        all_pw_col_jaccard_graph = nx.read_gpickle(result_dir+'col_sim.pkl')
    else:
        print('Computing pairwise col similarity')
        all_pairwise_col_jaccard = similarity.get_pairwise_similarity(dataset, similarity.compute_col_jaccard_DF)
        all_pw_col_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_col_jaccard)
        # Write out the Pairwise Distances as Adj list
        nx.to_pandas_adjacency(all_pw_col_jaccard_graph, weight='weight').to_csv(
            result_dir + 'col_sim.csv')
        nx.write_gpickle(all_pw_col_jaccard_graph, result_dir + 'col_sim.pkl')


    '''
    if os.path.exists(result_dir+'val_sim.pkl') and recompute:
        all_pw_val_jaccard_graph = nx.read_gpickle(result_dir+'val_sim.pkl')
    else:
        print('Computing pairwise val similarity')
        all_pairwise_val_jaccard = similarity.get_pairwise_similarity_valset(dataset)
        all_pw_val_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_val_jaccard)
        # Write out the Pairwise Distances as Adj list
        nx.to_pandas_adjacency(all_pw_val_jaccard_graph, weight='weight').to_csv(
            result_dir + 'val_sim.csv')
        nx.write_gpickle(all_pw_val_jaccard_graph, result_dir + 'val_sim.pkl')


    if os.path.exists(result_dir+'rowval_sim.pkl') and recompute:
        all_pw_ival_jaccard_graph = nx.read_gpickle(result_dir+'rowval_sim.pkl')
    else:
        print('Computing pairwise rowval similarity')
        all_pairwise_ival_jaccard = similarity.get_pairwise_similarity_valset(dataset, indexed=True)
        all_pw_ival_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_ival_jaccard)
        # Write out the Pairwise Distances as Adj list
        nx.to_pandas_adjacency(all_pw_ival_jaccard_graph, weight='weight').to_csv(
            result_dir + 'ival_sim.csv')
        nx.write_gpickle(all_pw_ival_jaccard_graph, result_dir + 'rowval_sim.pkl')


    if os.path.exists(result_dir+'colval_sim.pkl') and recompute:
        all_pw_colval_jaccard_graph = nx.read_gpickle(result_dir+'colval_sim.pkl')
    else:
        print('Computing pairwise colval similarity')
        all_pairwise_colval_jaccard = similarity.get_pairwise_similarity_colvalset(dataset)
        all_pw_colval_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_colval_jaccard)
        # Write out the Pairwise Distances as Adj list
        nx.to_pandas_adjacency(all_pw_colval_jaccard_graph, weight='weight').to_csv(
            result_dir + 'colval_sim.csv')
        nx.write_gpickle(all_pw_colval_jaccard_graph, result_dir + 'colval_sim.pkl')


    if os.path.exists(result_dir+'cellval_sim.pkl') and recompute:
        all_pw_cellval_jaccard_graph = nx.read_gpickle(result_dir+'cellval_sim.pkl')
    else:
        print('Computing pairwise cellval similarity')
        all_pairwise_cellval_jaccard = similarity.get_pairwise_similarity_cellvalset(dataset)
        all_pw_cellval_jaccard_graph = graphs.generate_pairwise_graph(all_pairwise_cellval_jaccard)
        # Write out the Pairwise Distances as Adj list
        nx.to_pandas_adjacency(all_pw_cellval_jaccard_graph, weight='weight').to_csv(
            result_dir + 'cellval_sim.csv')
        nx.write_gpickle(all_pw_cellval_jaccard_graph, result_dir + 'cellval_sim.pkl')

    '''

    if flip_sim:
        temp = all_pw_jaccard_graph
        all_pw_jaccard_graph = all_pw_col_jaccard_graph
        all_pw_col_jaccard_graph = temp
        temp = cell_threshold
        cell_threshold = col_threshold
        col_threshold  = temp


    # Check for files in the ground truth that are missing in file list
    missing_files = ds.check_csv_graph(artifact_dir, g_truth)

    # Cluster for visualization
    clusters = clustering.exact_schema_cluster(dataset)
    clustering.write_clusters_to_file(clusters, result_dir + 'clusters_with_filename.csv')
    cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')

    # Start with intra-cluster edges:
    if(pre_cluster != 'No Precluster'):
        pairwise_jaccard = precomputed_sim.intra_cluster_similarity_pc(dataset, clusters, all_pw_jaccard_graph, threshold=cell_threshold)
    else:
        pairwise_jaccard = precomputed_sim.get_pairwise_similarity_pc(dataset, all_pw_jaccard_graph, threshold=cell_threshold)

    pw_jaccard_graph = graphs.generate_pairwise_graph(pairwise_jaccard)

    if flip_sim:
        edge_t = 'col'
    else:
        edge_t = 'cell'

    g_inferred, edge_num = clustering.max_spanning_tree(pw_jaccard_graph, edge_type=edge_t)

    for edge in g_inferred.edges(data=True):
        print('Adding Intra-Cluster Tree Edge:', edge[0], edge[1], edge_t+"-level score:", edge[2]['weight'])

    # 02/02/2020
    # Start with empty graph to try to force spanning tree tie breaking via columnar containment
    # Doing this to try to reduce false positives at spanning tree stage.
    # g_inferred = nx.empty_graph()






    # Draw first graph and get results
    #nx.write_edgelist(g_inferred, result_dir + 'infered_mst_cell.csv', data=True)
    #if draw:
    #    img_frames.append(graphs.generate_and_draw_graph(base_dir, nb_name, 'cell', cluster_dict=cluster_dict,
    #                                                     join_list=None))

    use_col = 'col' in metric

    if pre_cluster == 'No Precluster':
        if metric == 'valset':
            jaccard_graph = all_pw_val_jaccard_graph
        elif metric == 'rowvalset':
            jaccard_graph = all_pw_ival_jaccard_graph
        elif metric == 'colvalset':
            jaccard_graph = all_pw_colval_jaccard_graph
        elif metric == 'cellvalset':
            jaccard_graph = all_pw_cellval_jaccard_graph
        else:
            jaccard_graph = all_pw_jaccard_graph

        threshold_graph = graphs.get_subgraph_threshold(jaccard_graph, cell_threshold)
        g_inferred, edge_num = clustering.max_spanning_tree(threshold_graph, edge_type=metric.split('+')[0])
        pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster,
                              timeit.default_timer() - start_time, metric=metric)

    # Now try adding columnar edges to each disconnected cluster subgraph:
    elif pre_cluster == 'PC2':
        g_inferred, edge_num = intra_cluster_add_col_edges(dataset, clusters, g_inferred, edge_num, all_pw_jaccard_graph, all_pw_col_jaccard_graph=all_pw_col_jaccard_graph, cell_threshold=cell_threshold, col_threshold=col_threshold, col=use_col)

        # Draw first graph and get results
        pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster,
                              timeit.default_timer() - start_time, metric=metric)
        if draw:
            img_frames.append(graphs.generate_and_draw_graph(base_dir, nb_name, 'cell',
                                                             cluster_dict=cluster_dict, join_list=None))

    print("Finished adding all intra-cluster edges")
    nx.write_edgelist(g_inferred, result_dir + 'infered_mst_'+metric+'.csv', data=True)


    # Add vertices to the graph if they don't exist
    for artifact in dataset.keys():
        if artifact not in [n for n in g_inferred.nodes()]:
            print('Adding artifact to graph', artifact)
            g_inferred.add_node(artifact)



    components = [c for c in nx.connected_components(g_inferred)]
    #print('Components:', components)

    steps = 0
    stop = False

    if pre_cluster == 'No Precluster':
        stop = True

    # Clustering Loop Starts here
    while (len(components) > 1 and steps < 20 and not stop):

        steps += 1

        new_graph, new_edge_num = clustering.find_components_join_edge(g_inferred, dataset, edge_num, pw_graph=all_pw_jaccard_graph, col_pw_graph=all_pw_col_jaccard_graph, cell_threshold=cell_threshold, col_threshold=col_threshold, col=use_col)
        if not new_graph:
            stop = True
        else:
            g_inferred, edge_num = new_graph, new_edge_num

        components = [c for c in nx.connected_components(g_inferred)]
        #print('Components:', components)

        #nx.write_edgelist(g_inferred, result_dir + 'infered_mst_cell.csv', data=True)
        nx.write_edgelist(g_inferred, result_dir + 'infered_mst_'+metric+'.csv', data=True)


        # Draw inferred graph image:
        if (len(clusters) > 1):
            clustering.write_clusters_to_file(clusters, result_dir + 'clusters_with_filename.csv')
            cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')
        else:
            cluster_dict = None

        pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster,
                              timeit.default_timer() - start_time, metric=metric)
        if draw:
            img_frames.append(graphs.generate_and_draw_graph(base_dir, nb_name, 'cell',
                                                             cluster_dict=cluster_dict, join_list=None))

    # Test for NPPOs:

    group_list = None

    if group_edges:
        print('In group checking')

        # NPPO Clustering Loop Starts here
        stop = False
        nppo_dict = {}
        while (len(components) > 1 and steps < 20 and not stop):
            steps += 1
            new_graph, new_edge_num, nppo_dict = nppo.find_components_nppo_edge(g_inferred, dataset, edge_num,
                                                                                nppo_dict)
            if not new_graph:
                stop = True
            else:
                g_inferred, edge_num = new_graph, new_edge_num

            components = [c for c in nx.connected_components(g_inferred)]

            nx.write_edgelist(g_inferred, result_dir + 'infered_mst_' + metric + '.csv', data=True)

            pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files,
                                  pre_cluster, timeit.default_timer() - start_time, metric=metric)

            # print(g_inferred.nodes(), g_inferred.edges())



    inferred_j_edges = []
    join_list = None
    cluster_dict = None

    if join_edges:
        print('Checking for Joins')

        # NPPO Clustering Loop Starts here
        stop = False
        while (len(components) > 1 and steps < 20 and not stop):
            steps += 1
            triple_dict = {}
            new_graph, new_edge_num, triple_dict = nppo.find_components_join_edge(g_inferred, dataset, edge_num, triple_dict)
            if not new_graph:
                stop = True
            else:
                g_inferred, edge_num = new_graph, new_edge_num

            components = [c for c in nx.connected_components(g_inferred)]

            nx.write_edgelist(g_inferred, result_dir + 'infered_mst_' + metric + '.csv', data=True)

            pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files,
                                  pre_cluster, timeit.default_timer() - start_time, metric=metric)

            # print(g_inferred.nodes(), g_inferred.edges())



        '''
        print("Adding Join Edges")
        join_list = nppo.find_all_joins_df_dict(dataset)
        print(len(join_list), "Joins Detected")
        '''
        #g_inferred = nppo.add_join_edges(join_list, g_inferred)

        #for join in join_list:
        #    inferred_j_edges.append((join[0], join[2]))
        #    inferred_j_edges.append((join[1], join[2]))

        #nppo.write_join_candidates(join_list, result_dir + 'join_candidates.csv')

        #g_truth_j_edges = [(u, v) for u, v, d in g_truth.edges(data=True) \
        #                   if g_truth[u][v]['operation'] == 'merge']

        # Check Join Precision/Recall
        # print(get_join_precision_recall(g_truth_j_edges, inferred_j_edges))

        #result = graphs.get_precision_recall(g_truth, g_inferred)

        #cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')
        #pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster,
        #                       timeit.default_timer() - start_time, metric=metric)
        #if draw:
        #    img_frames.append(graphs.generate_and_draw_graph(base_dir, nb_name, 'cell',
        #                                                     cluster_dict=cluster_dict, join_list=None))


    if transform_edges:
        print('In transform checking')

        # NPPO Clustering Loop Starts here
        stop = False
        nppo_dict = {}
        while (len(components) > 1 and steps < 20 and not stop):
            steps += 1
            new_graph, new_edge_num, nppo_dict = nppo.find_components_nppo_edge(g_inferred, dataset, edge_num,
                                                                                nppo_dict, nppo_function=nppo.transform_detector, label='transform')
            if not new_graph:
                stop = True
            else:
                g_inferred, edge_num = new_graph, new_edge_num

            components = [c for c in nx.connected_components(g_inferred)]

            nx.write_edgelist(g_inferred, result_dir + 'infered_mst_' + metric + '.csv', data=True)

            pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files,
                                  pre_cluster, timeit.default_timer() - start_time, metric=metric)

        '''
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        groupbys = nppo.get_all_groupbys_dfdict(dataset)
        group_list = [(x[0], x[2]) for x in groupbys]
        pp.pprint(groupbys)
        g_inferred = nppo.add_group_edges(group_list, g_inferred)

        # Check Group Precision/Recall

        result = graphs.get_precision_recall(g_truth, g_inferred)

        cluster_dict = clustering.get_graph_clusters(result_dir + 'clusters_with_filename.csv')
        pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files, pre_cluster,
                              timeit.default_timer() - start_time, metric=metric)
        if draw:
            img_frames.append(graphs.generate_and_draw_graph(base_dir, nb_name, 'cell',
                                                             cluster_dict=cluster_dict, join_list=None))

        '''

    if pivot_edges:
        print('In pivot checking')

        # NPPO Clustering Loop Starts here
        stop = False
        nppo_dict = {}
        while (len(components) > 1 and steps < 20 and not stop):
            steps += 1
            new_graph, new_edge_num, nppo_dict = nppo.find_components_nppo_edge(g_inferred, dataset, edge_num,
                                                                                nppo_dict, nppo_function=nppo.pivot_detector, label='pivot')
            if not new_graph:
                stop = True
            else:
                g_inferred, edge_num = new_graph, new_edge_num

            components = [c for c in nx.connected_components(g_inferred)]

            nx.write_edgelist(g_inferred, result_dir + 'infered_mst_' + metric + '.csv', data=True)

            pr_df = append_result(pr_df, dataset, g_truth, g_inferred, nb_name, index, clusters, missing_files,
                                  pre_cluster, timeit.default_timer() - start_time, metric=metric)





    image_frames = [Image.open(frame) for frame in img_frames]

    if draw:
        image_frames[0].save(base_dir+nb_name+'/'+metric+'_relic_construction.gif',
                             format='GIF', append_images=image_frames[1:],
                             save_all=True,
                             duration=1000,
                             loop=0)

    return pr_df, image_frames


# Note: Edges in g_inferred should only be within a cluster
def intra_cluster_add_col_edges(df_dict, clusters, g_inferred, edge_num, all_pw_jaccard_graph, all_pw_col_jaccard_graph=None, cell_threshold=0.1, col_threshold=0.1, col=True):

    for cluster in clusters.values():
        subgraph = g_inferred.subgraph(cluster).copy()
        components = [c for c in nx.connected_components(subgraph)]

        steps = 0
        stop = False
        limit = len(components) * len(components)

        # Clustering Loop Starts here
        while (len(components) > 1 and steps < limit and not stop):

            steps += 1

            new_graph ,new_edge_num = clustering.find_components_join_edge(subgraph, df_dict, edge_num, pw_graph=all_pw_jaccard_graph, col_pw_graph=all_pw_col_jaccard_graph, cell_threshold=cell_threshold, col_threshold=col_threshold, col=col)
            if not new_graph:
                stop = True
            else:
                subgraph = new_graph
                edge_num = new_edge_num

            components = [c for c in nx.connected_components(subgraph)]
            # print('Components:', components)


        for edge in subgraph.edges(data=True):
            g_inferred.add_edge(edge[0], edge[1], weight=edge[2]['weight'], num=edge[2]['num'], type=edge[2]['type'])
            #edge_num += 1


    return g_inferred, edge_num


def experiment_1(base_dir, nb_name, clustering, metric, swap, recompute, group, join, transform, pivot):
    pd.set_option('display.max_columns', None)

    if group:
        metric = metric+'+group'
    if join:
        metric = metric+'+join'
    if transform:
        metric = metric + '+transform'
    if pivot:
        metric = metric + '+pivot'

    result, im_frames = lineage_inference_agglomerative(base_dir=base_dir, nb_name=nb_name, pre_cluster=clustering, metric=metric, flip_sim=swap, recompute=recompute, group_edges=group, join_edges=join, transform_edges=transform, pivot_edges=pivot)
    print(result[['numclusters', 'edges_correct', 'edges_missing', 'edges_to_remove', 'F1', 'time']])
    result.to_csv(base_dir + nb_name + '/'+metric+'_relic_result.csv')

    return result

def experiment_2(base_dir, nb_name, clustering, metric, swap, recompute):
    pd.set_option('display.max_columns', None)

    exp2_df = pd.DataFrame()

    for cell_threshold in np.arange(0, 1.0, 0.05):
        for col_threshold  in np.arange(0, 1.0, 0.05):
            print('Thresholds', cell_threshold, col_threshold)
            result, im_frames = lineage_inference_agglomerative(base_dir=base_dir, nb_name=nb_name, pre_cluster=clustering, metric=metric, flip_sim=swap, recompute=recompute, cell_threshold=cell_threshold, col_threshold=col_threshold)
            print(result[['numclusters', 'edges_correct', 'edges_missing', 'edges_to_remove', 'F1', 'time']])
            result.to_csv(base_dir + nb_name + '/relic_agglomerative_result_'+str(cell_threshold)+'_'+ str(col_threshold)+'.csv')
            line = result.iloc[-1]
            line['cell_threshold'] = cell_threshold
            line['col_threshold'] = col_threshold
            exp2_df = exp2_df.append(line)

    exp2_df.to_csv(base_dir + nb_name + '/relic_thresholding_final_result.csv')

    return exp2_df

def experiment_3(base_dir, nb_name, clustering, metric, swap, recompute):
    pd.set_option('display.max_columns', None)

    exp2_df = pd.DataFrame()

    for cell_threshold in np.arange(0, 1.0, 0.05):
            print('Thresholds', cell_threshold)
            result, im_frames = lineage_inference_agglomerative(base_dir=base_dir, nb_name=nb_name, pre_cluster=clustering, metric=metric, flip_sim=swap, recompute=recompute, cell_threshold=cell_threshold, col_threshold=cell_threshold)
            print(result[['numclusters', 'edges_correct', 'edges_missing', 'edges_to_remove', 'F1', 'time']])
            result.to_csv(base_dir + nb_name + '/relic_agglomerative_result_'+metric+'_'+str(cell_threshold)+'_'+'.csv')
            line = result.iloc[-1]
            line['threshold'] = cell_threshold
            exp2_df = exp2_df.append(line)

    exp2_df.to_csv(base_dir + nb_name + '/relic_thresholding_final_result_'+metric+'.csv')

    return exp2_df

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
                        type=str, default='PC2')

    parser.add_argument("--metric",
                        help="Result metric name",
                        type=str, default='pc2cellcol')

    parser.add_argument("--swap",
                        help="Swap cell/col metric precedence",
                        type=bool, default=False)

    parser.add_argument("--recompute",
                        help="Use existing pairwise distance files",
                        type=bool, default=False)

    parser.add_argument("--group",
                        help="Use Group Edge Detection",
                        type=bool, default=False)

    parser.add_argument("--join",
                        help="Use Join Edge Detection",
                        type=bool, default=False)

    parser.add_argument("--transform",
                        help="Use Transform Edge Detection",
                        type=bool, default=False)

    parser.add_argument("--pivot",
                        help="Use Pivot Edge Detection",
                        type=bool, default=False)


    options = parser.parse_args(args)

    return options


def main(args=sys.argv[1:]):

    options = setup_arguments(args)
    #experiment_1(options.basedir, options.nbname, options.clustering)
    experiment_1(options.basedir, options.nbname, options.clustering, options.metric, options.swap, options.recompute, options.group, options.join, options.transform, options.pivot)
    #experiment_2(options.basedir, options.nbname, options.clustering, options.metric, options.swap, options.recompute)
    #experiment_3(options.basedir, options.nbname, options.clustering, options.metric, options.swap, options.recompute)




if __name__ == "__main__":
    main()
