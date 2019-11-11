import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter

import io

GRAPH_EDGE_ARGS = '-Eminlen=50.0'


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
        edge_labels = {i[0:2]: '{0:.2f}'.format(i[2]['weight'])
                       for i in G.edges(data=True)}
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

    to_add  = g_edge_set - t_edge_set
    to_remove = t_edge_set - g_edge_set

    try:
        precision = float(len(correct))/len(g_edge_set)
        recall = float(len(correct))/len(t_edge_set)
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
    result_file = base_dir+nb_name+'/'+nb_name+'_gt_fixed.pkl'
    return nx.read_gpickle(result_file)

def get_graph_edge_list(base_dir, nb_name, metric):
    result_file = base_dir+nb_name+'/inferred/infered_mst_'+metric+'.csv'
    #return nx.read_edgelist(result_file, delimiter=',', data=(('weight', float),))
    return nx.read_edgelist(result_file)

def get_distance_matrix(base_dir, nb_name, metric):
    result_file = base_dir+nb_name+'/inferred/'+metric+'_sim.csv'
    return pd.read_csv(result_file, index_col=0)


def generate_notebook_image(base_dir, nb_name, canvas_size=(30,30)):
    g_truth = get_graph(base_dir, nb_name)
    plt = draw_graph(g_truth, canvas_size=canvas_size, show=False)
    plt.savefig(base_dir+nb_name+'/'+nb_name+'_gt.png')
    plt.clf()
    return plt



def generate_explaination_graph(g_truth, g_inferred, distance_matrix):
    exp_graph = g_truth.copy()
    nx.set_edge_attributes(exp_graph, True, 'truth')
    for edge_t in g_inferred.edges(data=True):
        exp_graph.add_edge(edge_t[0], edge_t[1], inferred=True)
        if g_truth.has_edge(edge_t[0], edge_t[1]):
            exp_graph[edge_t[0]][edge_t[1]]['correct']=True
        elif g_truth.has_edge(edge_t[1], edge_t[0]):
            exp_graph[edge_t[0]][edge_t[1]]['correct']=True
        else:
            exp_graph[edge_t[0]][edge_t[1]]['correct']=False

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
    print(root)
    if root is None:
        print(G)
        root = sorted(nx.degree(G), key=lambda kv: kv[1])[0][0]
        print(root)

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
        node_color = [cluster_dict[e]/20 for e in G.nodes()]
        cmap = 'rainbow'
    else :
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

    #print(correct_edges)

    nx.draw_networkx_edges(G, pos, edgelist=correct_edges, width=8, alpha=0.5, edge_color='green', arrows=False)

    incorrect_edges = [edge for edge in G.edges(data=True)
                       if 'correct' in edge[2] and edge[2]['correct'] == False]

    nx.draw_networkx_edges(G, pos, edgelist=incorrect_edges, width=3, alpha=0.5, edge_color='red', style='dashed', connectionstyle='Arc3, rad=0.1', arrows=False)

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
                already_marked[(i[0],i[1])] = True

        if 'weight' in i[2]:
            edge_label = edge_label + ' ('+'{0:.8f}'.format(i[2]['weight'])+')'

        if (i[1],i[0]) not in already_marked.keys():
            edge_labels[(i[0],i[1])] = edge_label

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=16)

    return plt


def generate_and_draw_graph(base_dir, nb_name, metric, root=None, cluster_dict=None, join_list=None,
                            group_list=None):
    g_truth = get_graph(base_dir, nb_name)
    g_infered = get_graph_edge_list(base_dir, nb_name,metric)
    dist = get_distance_matrix(base_dir, nb_name,metric)
    exp_graph = generate_explaination_graph(g_truth, g_infered, dist)

    plt=draw_exp_graph(exp_graph, canvas_size=(50,50), g_truth=g_truth, root=root, cluster_dict=cluster_dict, join_list=join_list)

    figure_file = base_dir+nb_name+'/'+nb_name+'_inferred.png'
    if os.path.isfile(figure_file):
        print('Overwriting file: ' + figure_file)
        os.remove(figure_file)
    plt.savefig(figure_file)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    #plt.clf()
    plt.show()
    return buf
