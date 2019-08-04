import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from collections import Counter



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
def draw_graph(G, canvas_size=(8, 12), node_size=1500,
               layout_fn=graphviz_layout, **kwargs):
    # Set Canvas Size
    plt.figure(10, figsize=canvas_size)

    # Remove axes and set a margin to prevent cut-off
    ax = plt.gca()
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.30)

    root = kwargs.pop('root', None)
    pos = layout_fn(G, root=root, prog='dot')

    try:
        edge_labels = {i[0:2]: '{0:.2f}'.format(i[2]['weight'])
                       for i in G.edges(data=True)}
    except:
        edge_labels = None

    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=node_size)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()


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