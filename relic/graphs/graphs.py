import networkx as nx
import os

from pyvis.network import Network
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

import io

from relic.distance import ppo
from relic.graphs import clustering

import numpy as np

from relic.utils.serialize import build_df_dict_dir
import logging

logger = logging.getLogger(__name__)

GRAPH_EDGE_ARGS = '-Eminlen=1000.0'


# Generates a weighted graph of pairwise similarity scores
# excepts a list of node1, node2, score tuples
def generate_pairwise_graph(pairwise_similarity):
    G = nx.Graph()
    for node1, node2, sim in pairwise_similarity:
        G.add_edge(node1, node2, weight=sim)
    return G


# Generates Spanning Tree from Weighted Graph:
def generate_spanning_tree(G):
    return nx.maximum_spanning_tree(G)


def set_graph_weights(G, dataset, similarity_metric):
    weights = {}
    for node1, node2 in G.edges:
        weights[(node1, node2)] = \
            similarity_metric(dataset[node1], dataset[node2])

    nx.set_edge_attributes(G, name='weight', values=weights)
    return G


# Draw Graph Plot:
def draw_graph(G, canvas_size=(8, 12), node_size=2000,
               layout_fn=graphviz_layout, show=True, **kwargs):
    # Set Canvas Size
    plt.figure(10, figsize=canvas_size)

    # Remove axes and set a margin to prevent cut-off
    ax = plt.gca()
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.30)

    root = kwargs.pop('root', None)
    pos = layout_fn(G, root=root, prog='dot', args=GRAPH_EDGE_ARGS)

    try:
        # edge_labels = {i[0:2]: '{0:.2f}'.format(i[2]['weight'])
        #               for i in G.edges(data=True)}
        edge_labels = {e[0:2]: e[2]['operation'] for e in G.edges(data=True)}
    except:
        edge_labels = None

    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=node_size, font_size=20)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=16)
    if show:
        plt.show()

    return plt


def get_precision_recall(G_truth, T_inferred):
    g_edge_set = set([frozenset((v1, v2)) for v1, v2 in G_truth.edges])
    t_edge_set = set([frozenset((v1, v2)) for v1, v2 in T_inferred.edges])

    correct = g_edge_set.intersection(t_edge_set)

    to_add = g_edge_set - t_edge_set
    to_remove = t_edge_set - g_edge_set

    try:
        precision = float(len(correct)) / len(t_edge_set)
        recall = float(len(correct)) / len(g_edge_set)
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError as e:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    return {'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'correct_edges': correct,
            'to_add': to_add,
            'to_remove': to_remove}


def load_dataset_list(dslist):
    return {str(k): v for k, v in enumerate(dslist)}


def get_graph(base_dir, nb_name):
    result_file = base_dir + nb_name + '/' + nb_name + '_gt_fixed.pkl'
    return nx.read_gpickle(result_file)


def get_graph_edge_list(base_dir, nb_name, metric):
    result_file = base_dir + nb_name + '/inferred/infered_mst_' + metric + '.csv'
    # return nx.read_edgelist(result_file, delimiter=',', data=(('weight', float),))
    return nx.read_edgelist(result_file)


def get_distance_matrix(base_dir, nb_name, metric):
    result_file = base_dir + nb_name + '/inferred/' + metric + '_sim.csv'
    return pd.read_csv(result_file, index_col=0)


def generate_notebook_image(base_dir, nb_name, canvas_size=(50, 50)):
    g_truth = get_graph(base_dir, nb_name)
    plt = draw_graph(g_truth, canvas_size=canvas_size, show=False)
    plt.savefig(base_dir + nb_name + '/' + nb_name + '_gt.png')
    plt.clf()
    return plt


def generate_explaination_graph(g_truth, g_inferred, distance_matrix):
    exp_graph = g_truth.copy()
    nx.set_edge_attributes(exp_graph, True, 'truth')
    for edge_t in g_inferred.edges(data=True):
        exp_graph.add_edge(edge_t[0], edge_t[1], inferred=True)
        if g_truth.has_edge(edge_t[0], edge_t[1]):
            exp_graph[edge_t[0]][edge_t[1]]['correct'] = True
        elif g_truth.has_edge(edge_t[1], edge_t[0]):
            exp_graph[edge_t[0]][edge_t[1]]['correct'] = True
        else:
            exp_graph[edge_t[0]][edge_t[1]]['correct'] = False

    for edge_t in exp_graph.edges():
        try:
            exp_graph[edge_t[0]][edge_t[1]]['weight'] = distance_matrix[edge_t[0]][edge_t[1]]
        except KeyError as e:
            exp_graph[edge_t[0]][edge_t[1]]['weight'] = 0.0

    return exp_graph


def draw_exp_graph(G, canvas_size=(30, 30), node_size=2000,
                   layout_fn=graphviz_layout, g_truth=None,
                   cluster_dict=None, join_list=None, group_list=None, **kwargs):
    # Set Canvas Size
    plt.figure(10, figsize=canvas_size)

    # Remove axes and set a margin to prevent cut-off
    ax = plt.gca()
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.30)

    root = kwargs.pop('root', None)
    # print(root)
    if root is None:
        # print(G)
        root = sorted(nx.degree(G), key=lambda kv: kv[1])[0][0]
        # print(root)

    if g_truth:
        pos = layout_fn(g_truth, root=root, prog='dot', args=GRAPH_EDGE_ARGS)
    else:
        pos = layout_fn(G, root=root, prog='dot', args=GRAPH_EDGE_ARGS)

    try:
        edge_labels = {i[0:2]: '{0:.4f}'.format(i[2]['weight'])
                       for i in G.edges(data=True)}
    except:
        edge_labels = None

    if cluster_dict:
        node_color = [cluster_dict[e] / 20 for e in G.nodes()]
        cmap = 'rainbow'
    else:
        node_color = 'r'
        cmap = None
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color,
                           node_size=500,
                           alpha=0.9,
                           cmap=cmap)
    nx.draw_networkx_labels(G, pos, font_size=20)

    correct_edges = [edge for edge in G.edges(data=True)
                     if 'correct' in edge[2] and edge[2]['correct']]

    # print(correct_edges)

    nx.draw_networkx_edges(G, pos, edgelist=correct_edges, width=8, alpha=0.5, edge_color='green', arrows=False)

    incorrect_edges = [edge for edge in G.edges(data=True)
                       if 'correct' in edge[2] and edge[2]['correct'] == False]

    nx.draw_networkx_edges(G, pos, edgelist=incorrect_edges, width=3, alpha=0.5, edge_color='red', style='dashed',
                           connectionstyle='Arc3, rad=0.1', arrows=False)

    # print(incorrect_edges)

    if join_list:
        nx.draw_networkx_edges(G, pos,
                               edgelist=join_list,
                               width=3, alpha=0.7, edge_color='purple', style='dotted',
                               arrows=False)

    if group_list:
        nx.draw_networkx_edges(G, pos,
                               edgelist=group_list,
                               width=4, alpha=0.7, edge_color='cyan', style='dashed',
                               arrows=False)

    gt_edges = [edge for edge in G.edges(data=True)
                if 'truth' in edge[2] and edge[2]['truth'] == True]

    nx.draw_networkx_edges(G, pos, edgelist=gt_edges, width=2, alpha=0.75, edge_color='black', arrows=True)

    already_marked = {}
    edge_labels = {}
    for i in G.edges(data=True):
        edge_label = ''
        if 'truth' in i[2] and i[2]['truth'] == True:
            if i[2]['operation'] is not None:
                edge_label = edge_label + i[2]['operation']
                already_marked[(i[0], i[1])] = True

        if 'weight' in i[2]:
            edge_label = edge_label + ' (' + '{0:.8f}'.format(i[2]['weight']) + ')'

        if (i[1], i[0]) not in already_marked.keys():
            edge_labels[(i[0], i[1])] = edge_label

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=16)

    return plt


def generate_and_draw_graph(base_dir, nb_name, metric, root=None, cluster_dict=None, join_list=None,
                            group_list=None):
    g_truth = get_graph(base_dir, nb_name)
    g_infered = get_graph_edge_list(base_dir, nb_name, metric)
    dist = get_distance_matrix(base_dir, nb_name, metric)
    exp_graph = generate_explaination_graph(g_truth, g_infered, dist)

    plt = draw_exp_graph(exp_graph, canvas_size=(100, 100), g_truth=g_truth, root=root, cluster_dict=cluster_dict,
                         join_list=join_list)

    figure_file = base_dir + nb_name + '/' + nb_name + '_inferred.png'
    if os.path.isfile(figure_file):
        # print('Overwriting file: ' + figure_file)
        os.remove(figure_file)
    plt.savefig(figure_file)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.clf()
    # plt.show()
    return buf


# Given a specific workflow graph with edge weights, prune edges below threshold
def get_subgraph_threshold(graph, threshold):
    g_copy = graph.copy()
    to_remove = [(u, v) for u, v in graph.edges if graph[u][v]['weight'] < threshold]

    for u, v in to_remove:
        g_copy.remove_edge(u, v)

    return g_copy


### New pyviz Graph Generators

def get_edge_color(e1, e2, g_truth, g_inferred):
    in_g_truth = False
    in_inferred = False
    if g_truth.has_edge(e1, e2) or g_truth.has_edge(e2, e1):
        in_g_truth = True
    if g_inferred.has_edge(e1, e2) or g_inferred.has_edge(e2, e1):
        in_inferred = True

    if in_g_truth and in_inferred:
        return 'green'
    elif in_inferred:
        return 'red'
    elif in_g_truth:
        return 'black'
    else:
        return '#D3D3D3'


def get_edge_number(e1, e2, g_inferred):
    if g_inferred.has_edge(e1, e2) and 'num' in g_inferred[e1][e2]:
        return g_inferred[e1][e2]['num']
    elif g_inferred.has_edge(e2, e1) and 'num' in g_inferred[e2][e1]:
        return g_inferred[e2][e1]['num']
    return None


def draw_interactive_graph(RESULT_DIR, selected_nb, metric='cell', weight='cell_jaccard', cached=False):
    # , bgcolor="#222222", font_color="white",
    nb_net = Network(height="750px", width="100%", notebook=True)

    g = get_graph(RESULT_DIR, selected_nb)  # .to_undirected()
    g_inferred = get_graph_edge_list(RESULT_DIR, selected_nb, metric)
    df_dict = ppo.load_dataset_dir(RESULT_DIR + selected_nb + '/artifacts/', '*.csv', index_col=0)

    if os.path.exists(RESULT_DIR + selected_nb + '/inferred/' + weight + '_sim.pkl') and cached:
        nb_data = nx.read_gpickle(RESULT_DIR + selected_nb + '/inferred/' + weight + '_sim.pkl')
    else:
        nb_data = pd.DataFrame(ppo.get_all_node_pair_scores(df_dict, g))

    # print(df_dict)

    if '0.csv' not in df_dict:
        try:
            root_node = [x for x in nx.topological_sort(g)][0]  # TODO: Check more than one root issues
        except nx.NetworkXError as e:
            print("ERROR: Cycle in Graph")
            indeg = g.in_degree()
            root_node = [n for n in indeg if indeg[n] == 0][0]
            pass
    else:
        root_node = '0.csv'

    pos = graphviz_layout(g, root=root_node, prog='dot', args=GRAPH_EDGE_ARGS)

    # Cluster Coloring
    cluster_dict = clustering.get_graph_clusters(RESULT_DIR + selected_nb + '/inferred/' + 'clusters_with_filename.csv')

    cmap = plt.cm.Dark2(np.linspace(0, 1, len(set(cluster_dict.values()))))
    node_color = {e: to_hex(cmap[cluster_dict[e]]) for e in g.nodes()}

    if not cached:
        sources = nb_data['source']
        targets = nb_data['dest']
        weights = nb_data[weight]

        edge_data = zip(sources, targets, {'weight': w for w in weights})

    else:
        edge_data = nb_data.edges(data=True)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]['weight']

        # Edge Coloring
        edge_color = get_edge_color(e[0], e[1], g, g_inferred)

        edge_number = get_edge_number(e[0], e[1], g_inferred)

        if not cached:
            hover_dict = nb_data.loc[(nb_data.source == src) & (nb_data.dest == dst)].to_dict('records')[0]
        else:
            hover_dict = e[2]

        hover_string = "<br>".join([str(k) + " : " + str(v) for k, v in hover_dict.items()])

        src_node_hover_html = "Rows:" + str(len(df_dict[src])) + " Columns:" + str(len(set(df_dict[src]))) + \
                              "<br>" + df_dict[src].head().to_html()
        dst_node_hover_html = "Rows:" + str(len(df_dict[dst])) + " Columns:" + str(len(set(df_dict[dst]))) + \
                              "<br>" + df_dict[dst].head().to_html()
        nb_net.add_node(src, src, x=pos[src][0], y=pos[src][1], physics=False, title=src_node_hover_html,
                        color=node_color[src])
        nb_net.add_node(dst, dst, x=pos[dst][0], y=pos[dst][1], physics=False, title=dst_node_hover_html,
                        color=node_color[dst])

        # Ground Truth Operation Label:
        if g.to_undirected().has_edge(src, dst):
            hover_string += '<br> Operation: ' + str(g.to_undirected()[src][dst]['operation'])
            if 'args' in g.to_undirected()[src][dst]:
                hover_string += '<br> Args: ' + str(g.to_undirected()[src][dst]['args'])

        # Edge Coloring
        if edge_number is not None and 'type' in g_inferred[src][dst]:
            hover_string += '<br> Edge Type: ' + g_inferred[src][dst]['type']
            hover_string += ', score: {:.3f}'.format(g_inferred[src][dst]['weight'])
            nb_net.add_edge(src, dst, value=w, title=hover_string, physics=False, color=edge_color, label=edge_number)

        else:
            if edge_color != '#D3D3D3':  # Hack to remove TP edges
                nb_net.add_edge(src, dst, value=w, title=hover_string, physics=False, color=edge_color)

    return nb_net


def draw_web_graph(g_inferred, artifact_dir, inferred_dir, g_truth=None, width=1024, height=768):

    logger.debug(f"Canvas: {str(height)+'px'} X {str(width)+'px'}")
    nb_net = Network(height=str(height)+'px', width={str(width)+'px'})
    df_dict = build_df_dict_dir(artifact_dir)
    logger.debug(f'Inferred Graph: {g_inferred.nodes()}')

    if '0.csv' not in df_dict:
        if g_truth:
            zero_degree_nodes = [n for n, d in g_truth.in_degree() if d == 0]
            if zero_degree_nodes:
                root_node = zero_degree_nodes[0]
            else:
                root_node = np.random.choice([x for x in df_dict.keys()], 1)[0]
    else:
        root_node = '0.csv'

    logger.debug(f'Root Node: {root_node}')

    positional_graph = g_truth if g_truth else g_inferred
    pos = graphviz_layout(positional_graph, root=root_node, prog='dot')

    #logger.debug(f'Position Matrix: {pos}')

    # Cluster Coloring
    if os.path.exists(inferred_dir+'/clusters.txt'):
        cluster_dict = clustering.get_graph_clusters(inferred_dir+'/clusters.txt')
        cmap = plt.cm.Dark2(np.linspace(0, 1, len(set(cluster_dict.values()))))
        node_color = {e: to_hex(cmap[cluster_dict[e]]) for e in df_dict.keys()}
    else:
        node_color = {e: 'grey' for e in df_dict.keys()}

    all_edges = g_inferred.copy()
    if g_truth:
        all_edges = nx.compose(g_inferred, g_truth)

    for src, dst, data in all_edges.edges(data=True):
        w = data['weight'] if 'weight' in data else 0

        # Edge Coloring
        if g_truth:
            edge_color = get_edge_color(src, dst, g_truth, g_inferred)
        else:
            edge_color = 'lightgrey'

        edge_number = data['num'] if 'num' in data else 'x'

        hover_string = "<br>".join([str(k) + " : " + str(v) for k, v in data.items()])
        src_node_hover_html = "Rows:" + str(len(df_dict[src])) + " Columns:" + str(len(set(df_dict[src]))) + \
                              "<br>" + "(Click to Inspect Artifact)"
        dst_node_hover_html = "Rows:" + str(len(df_dict[dst])) + " Columns:" + str(len(set(df_dict[dst]))) + \
                              "<br>" + "(Click to Inspect Artifact)"
        nb_net.add_node(src, src, x=pos[src][0] - 150, y=pos[src][1] - 100, physics=False, title=src_node_hover_html,
                        color=node_color[src])
        nb_net.add_node(dst, dst, x=pos[dst][0] - 150, y=pos[dst][1] - 100, physics=False, title=dst_node_hover_html,
                        color=node_color[dst])

        # Ground Truth Operation Label:
        if g_truth and g_truth.to_undirected().has_edge(src, dst):
            hover_string += '<br>Ground Truth Operation: ' + str(g_truth.to_undirected()[src][dst]['operation'])
            if 'args' in g_truth.to_undirected()[src][dst]:
                hover_string += '<br>Generating Args: ' + str(g_truth.to_undirected()[src][dst]['args'])

        # Edge Coloring
        if g_inferred.has_edge(src, dst):
            hover_string += '<br> Edge Type: ' + g_inferred[src][dst]['type']
            hover_string += ', score: {:.3f}'.format(g_inferred[src][dst]['weight'])

        nb_net.add_edge(src, dst, value=w, title=hover_string, physics=False, label=edge_number,
                        color=edge_color)


    return nb_net
