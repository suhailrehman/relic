# import re

import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


class LineageTracker:

    def __init__(self, directory):
        self.num_files = 0
        self.graph = nx.DiGraph()
        self.directory = directory

    def item_exists(self, label):
        return label in self.graph.nodes

    def new_item(self, label, dataframe=None):
        if self.item_exists(label):
            raise NameError(label)
        self.graph.add_node(label)
        if type(dataframe) != 'NoneType':
            dataframe.to_csv(self.directory+label+'.csv',
                             sep=',')
        self.num_files += 1

    def link(self, src_label, dest_label, op_label, args=None):
        if not self.item_exists(src_label):
            raise NameError(src_label)
        if not self.item_exists(dest_label):
            raise NameError(dest_label)

        self.graph.add_edge(src_label,
                            dest_label,
                            operation=op_label, args=args)

    def draw_graph(self, canvas_size=(8, 12), node_size=1500,
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
        if root is not None:
            pos = layout_fn(self.graph, root=root, prog='dot')

        else:
            pos = layout_fn(self.graph, root=0)
        try:
            edge_labels = {
                            (i[0], i[1]): i[2]['operation']
                            for i in self.graph.edges(data=True)
                          }
        except Exception:
            edge_labels = None

        nx.draw_networkx(self.graph, pos=pos,
                         with_labels=True,
                         node_size=node_size)
        nx.draw_networkx_edge_labels(self.graph,
                                     pos=pos,
                                     edge_labels=edge_labels)

        plt.show()
