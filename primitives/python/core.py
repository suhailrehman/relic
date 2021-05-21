import dataset as ds
import clustering
from lineage import similarity, graphs  # , precomputed_sim
import nppo

import pandas as pd
import numpy as np
import os
import itertools
from collections import defaultdict
import logging
import networkx as nx
from networkx.utils import UnionFind


module_logger = logging.getLogger('relic.core')


class RelicAlgorithm:

    def __init__(self, input_dir, output_dir, name='wf_', **kwargs):
        # Logging Setup
        self.logger = logging.getLogger('relic.core.RelicAlgorithm')
        self.logger.info('Starting instance of RelicAlgorithm on %s', nb_name)

        # Directory Setup
        self.nb_name = name
        self.artifact_dir = input_dir
        self.inferred_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the dataset
        # TODO: Change to load/read on demand infrastructure
        self.dataset = ds.build_df_dict_dir(self.artifact_dir)

        # Create the initial graph
        self.g_inferred = nx.Graph()
        self.create_initial_graph()

        # Create initial components list and clustering
        self.components = UnionFind()
        self.initial_cluster = {}
        self.cluster_lookup = {}

        # Create the pairwise weights_dict
        # TODO: store multiple weights
        # TODO: Priority queue or self-sorting datastructure in association with unionfind
        #self.weight_df = None
        #self.set_weight_df()
        #self.weight_dict = defaultdict(dict)

        # Load the Ground Truth
        # TODO: Optional GT annotation or remove entirely
        # self.g_truth = graphs.get_graph(self.base_dir, self.nb_name).to_undirected()

        # Current Edge being added
        self.edge_no = 0

        # Tie Breaker Info
        self.tied_edges = {}
        self.two_tie_edges = {}

        # TODO : Instantenous Precision/Recall/F1 and other accuracy score by calling a single function
        # TODO : Instantaneous Graph

    def create_initial_graph(self):
        self.logger.debug('Creating the initial graph of artifact nodes')
        self.g_inferred = nx.Graph()
        for artifact in self.dataset.keys():
            if artifact not in [n for n in self.g_inferred.nodes()]:
                self.logger.debug('Adding artifact to graph %s', str(artifact))
                self.g_inferred.add_node(artifact)

        return self.g_inferred

    def set_initial_clusters(self, cluster_type='exact_schema'):
        # Initial Clusters set to individual artifacts:
        if cluster_type == 'exact_schema':
            self.initial_cluster = clustering.exact_schema_cluster(self.dataset)

        self.cluster_lookup = clustering.reverse_schema_dict(self.initial_cluster)

        return self.initial_cluster


    def initialize_components(self):
        # Initializes the current list of graph components:
        self.logger.debug('Updating Graph Components')
        self.components = UnionFind(elements=[n for c in nx.connected_components(self.g_inferred) for n in c])
        return self.components

    def compute_inter_component_weights(self, components=self.components, distance_function=similarity.compute_jaccard_label,
                                    weight_label='cell_jaccard', ntuples=2):
        ''' if weight_label not in self.weight_df:
            self.weight_df[weight_label] = np.nan '''
        if not components:
            self.update_components()
        for src_nodes, dst_nodes in itertools.combinations(components, ntuples):
            self.compute_pairs_distances(src_nodes, dst_nodes, distance_function, weight_label=weight_label)
        return self.weight_df

    def compute_intra_cluster_edges(self, distance_function=nppo.df_groupby_check_new,
                                    weight_label='groupby',
                                    cluster_dict=None):
        if not cluster_dict:
            cluster_dict = self.initial_cluster
        if weight_label not in self.weight_df:
            self.weight_df[weight_label] = np.nan
        for label, cluster_nodes in cluster_dict.items():
            if len(cluster_nodes) > 1:
                self.logger.debug('Elements in Cluster: %s', cluster_nodes)
                self.compute_pairs_distances(cluster_nodes, cluster_nodes,
                                             distance_function, weight_label=weight_label)
        return self.weight_df

    def compute_pairs_distances(self, src_nodes, dst_nodes, distance_function, weight_label, cached=True):
        pairs = list(itertools.product(src_nodes, dst_nodes))
        for u, v in pairs:
            if u == v:
                continue

            key = frozenset((u, v))
            self.check_edge_pair(key)

            if cached and pd.notna(self.weight_df.at[key, weight_label]):
                self.logger.debug('Weight %s already set for key %s', weight_label, key)
            else:
                score = distance_function(u, v, self.dataset, self.g_inferred)
                self.weight_df.at[key, weight_label] = score
                self.logger.debug('Edge %s has %s score of %s', key, weight_label, score)





    def compute_component_pairs(self):
        component_test = lambda x: self.components[list(x.name)[0]] != self.components[list(x.name)[1]]
        component_label = lambda x: str(self.components[list(x.name)[0]]) + str2(self.components[list(x.name)[1]])
        self.weight_df['cross_comp'] = self.weight_df.apply(component_test, axis=1)
        self.weight_df['cross_comp_label'] = self.weight_df.apply(component_label, axis=1)
        return self.weight_df

    def compute_component_link_edges(self, weight_label='groupby', threshold=1.0):
        df = self.weight_df
        cross_comp_edges = df.loc[(df.cross_comp == True) & (df[weight_label] >= threshold)]
        cross_comp_max = cross_comp_edges.groupby('cross_comp_label')[weight_label].transform(max)
        return cross_comp_max

    def add_edges_and_merge_components(self, edge, data):
        self.g_inferred.add_edge(edge[0], edge[1], **data)
        self.components.union(edge[0], edge[1])
        self.logger.debug(self.components)
        self.edge_no += 1

    def infer_link_edges(self, weight_label='groupby', threshold=1.0, max_step=None):
        self.compute_component_pairs()
        cross_comp_max = self.compute_component_link_edges(weight_label=weight_label, threshold=threshold)

        steps = 0
        if not max_step:
            max_step = len(self.dataset)
        while not cross_comp_max.empty and steps < max_step:
            sorted_list = cross_comp_max.sort_values(ascending=False)
            edge, weight = self.tie_break_max_score(sorted_list)  # sorted_list.index[0], sorted_list[0]

            self.logger.info('Adding edge #%d: %s, type %s, weight %f', self.edge_no, list(edge), weight_label, weight)
            self.add_edges_and_merge_components(list(edge), {'weight': weight,
                                                             'type': weight_label,
                                                             'num': self.edge_no})
            self.compute_component_pairs()
            cross_comp_max = self.compute_component_link_edges(weight_label=weight_label, threshold=threshold)
            self.logger.debug('Component List: %s', cross_comp_max)
            steps += 1


    '''
    Tie Breaking Functions
    '''
    def tie_break_max_score(self, sorted_cross_comp_max, tb_label='contraction_ratio'):
        max_score = sorted_cross_comp_max[0]
        tied_edges = sorted_cross_comp_max[sorted_cross_comp_max == max_score]
        if len(tied_edges) > 1:
            self.logger.debug('%d tied candidates for edge number %d', len(tied_edges), self.edge_no)
            self.tied_edges[self.edge_no] = tied_edges
            self.compute_tie_breaker_scores(tied_edges.index, tb_label=tb_label)
            scored_tb_edges = self.weight_df.loc[tied_edges.index].sort_values(tb_label, ascending=False)
            max_tb_score = scored_tb_edges.iloc[0][tb_label]
            two_tie_edges = scored_tb_edges[scored_tb_edges == max_tb_score]
            if len(two_tie_edges) > 1:
                self.logger.warning('%d TieBreaker Candidates for edge %d', len(two_tie_edges), self.edge_no)
                self.two_tie_edges[self.edge_no] = two_tie_edges[tb_label]

            return two_tie_edges.iloc[0].name, two_tie_edges.iloc[0][tb_label]

        else:
            return sorted_cross_comp_max.index[0], sorted_cross_comp_max[0]

    def compute_tie_breaker_scores(self, tied_list, tb_function=nppo.compute_contraction_ratio, tb_label='contraction_ratio'):
        for key in tied_list:
            score = tb_function(self.dataset, list(key)[0], list(key)[1])
            self.weight_df.at[key, tb_label] = score
        return self.weight_df


    '''
    Weight DF Methods 
    '''

    def set_weight_df(self):
        index = [frozenset((u, v)) for u, v in itertools.combinations(self.dataset.keys(), 2)]
        self.weight_df = pd.DataFrame(index=index)
        self.weight_df['nb'] = self.nb_name
        self.weight_df['artifacts'] = len(self.dataset)
        #self.weight_df['rows'] = len(self.dataset['0.csv'].index)
        #self.weight_df['columns'] = len(self.dataset['0.csv'])
        return self.weight_df


    def check_edge_pair(self, key):
        if key in self.weight_df.index:
            return True
        else:
            self.logger.warning('Edge %s not present in Weight DF', key)
            return False


    def augment_ground_truth(self):
        self.weight_df['ground_truth'] = False
        #self.weight_df['operation'] = np.nan

        for u, v, data in self.g_truth.edges(data=True):
            key = frozenset((u, v))
            if self.check_edge_pair(key):
                self.weight_df.at[key, 'ground_truth'] = True
                self.weight_df.at[key, 'operation'] = data['operation']

        return self.weight_df


    def augment_inferred_graph(self):
        self.weight_df['g_inferred'] = False
        #self.weight_df['operation'] = np.nan

        for u, v, data in self.g_inferred.edges(data=True):
            key = frozenset((u, v))
            if self.check_edge_pair(key):
                self.weight_df.at[key, 'g_inferred'] = True
                self.weight_df.at[key, 'type'] = data['type']

        return self.weight_df

    def augment_cluster_info(self):
        cluster_result = []

        for ix in self.weight_df.index:
            edge_pair = list(ix)
            cluster_result.append(self.cluster_lookup[edge_pair[0]] == self.cluster_lookup[edge_pair[1]])

        self.weight_df['same_cluster'] = pd.Series(cluster_result, index=self.weight_df.index)
        return self.weight_df
