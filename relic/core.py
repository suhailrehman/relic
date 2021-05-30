import pandas as pd
import numpy as np
import os
import itertools
from collections import defaultdict
import logging
import networkx as nx
from networkx.utils import UnionFind

from relic.distance.ppo import compute_all_ppo
from relic.distance.tiebreakers import tiebreak_hash_edge
from relic.graphs.clustering import exact_schema_cluster, reverse_schema_dict
from relic.utils.pqedge import PQEdges
from relic.utils.serialize import build_df_dict_dir
from relic.algorithm import compute_tuplewise_similarity
from relic.utils.artifactdict import ArtifactDict

module_logger = logging.getLogger('relic.core')


class RelicAlgorithm:

    def __init__(self, input_dir, output_dir, name='wf_', **kwargs):
        # Logging Setup
        self.logger = logging.getLogger('relic.core.RelicAlgorithm')
        self.logger.info('Starting instance of RelicAlgorithm on %s', name)

        # Directory Setup
        self.nb_name = name
        self.artifact_dir = input_dir
        self.inferred_dir = output_dir
        os.makedirs(self.inferred_dir, exist_ok=True)

        # Load the dataset
        # TODO: Change to load/read on demand infrastructure
        #self.dataset = build_df_dict_dir(self.artifact_dir)
        self.dataset = ArtifactDict(self.artifact_dir)

        # Create the initial graph
        self.g_inferred = nx.Graph()
        self.create_initial_graph()

        # Create initial components list and clustering
        self.components = UnionFind()
        self.initial_cluster = {}
        self.cluster_lookup = {}

        # Create the pairwise weights_dict
        # store multiple weights
        # Priority queue or self-sorting datastructure in association with unionfind
        self.pairwise_weights = defaultdict(PQEdges)
        # self.weight_df = None
        # self.set_weight_df()
        # self.weight_dict = defaultdict(dict)

        # Load the Ground Truth
        # TODO: Optional GT annotation or remove entirely
        # self.g_truth = graphs.get_graph(self.base_dir, self.nb_name).to_undirected()

        # Current Edge being added
        self.edge_no = 0

        # Tie Breaker Info
        self.tied_edges = {}
        self.two_tie_edges = {}

        # TODO : Instantaneous Precision/Recall/F1 and other accuracy score by calling a single function
        # TODO : Instantaneous Graph

    def load_artifacts(self):
        pass

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
            self.initial_cluster = exact_schema_cluster(self.dataset)
        self.cluster_lookup = reverse_schema_dict(self.initial_cluster)

        return self.initial_cluster

    def update_components(self):
        # Initializes the current list of graph components:
        self.logger.debug('Updating Graph Components')
        components = [n for c in nx.connected_components(self.g_inferred) for n in c]
        self.components = UnionFind(elements=components)
        self.num_components = len(components)
        return self.components

    def add_edge_and_merge_components(self, edge, data):
        self.g_inferred.add_edge(edge[0], edge[1], **data)
        self.components.union(edge[0], edge[1])
        self.logger.debug(self.components)
        self.edge_no += 1
        self.num_components -= 1

    def add_edges_of_type(self, edge_type='jaccard', similarity_function=compute_all_ppo,
                          tie_break_type='overlap', tie_break_function=None, final_function=None,
                          sim_threshold=0.1, tie_break_threshold=0.1, n_pairs=2, max_edges=None,
                          max_step=False):

        if not max_edges:
            max_edges = len(self.dataset)

        # Compute all pairwise edges first
        # TODO: Add custom pairs function for join and other detectors
        if edge_type not in self.pairwise_weights:
            self.pairwise_weights = compute_tuplewise_similarity(self.dataset, similarity_metric=similarity_function,
                                                                 threshold=sim_threshold, n_pairs=n_pairs)
        self.update_components()
        steps = 0
        stop = False
        if not max_step:
            max_step = len(self.dataset)

        while self.num_components > 1 and steps < max_step and not stop:
            max_edges = self.pairwise_weights[edge_type].pop_max()
            if len(max_edges) > 1:
                tie_break_edges = tie_break_function(max_edges)
                if len(tie_break_edges) > 1:
                    edge = tiebreak_hash_edge(tie_break_edges)
                else:
                    edge = max_edges[0]
            else:
                edge = max_edges[0]

            weight = edge[1]

            self.logger.info('Adding edge #%d: %s, type %s', self.edge_no, list(edge), edge_type)
            self.add_edges_and_merge_components(list(edge), {'weight': weight,
                                                             'type': edge_type,
                                                             'num': self.edge_no})
            '''
            self.compute_component_pairs()
            cross_comp_max = self.compute_component_link_edges(weight_label=edge_type, threshold=sim_threshold)
            self.logger.debug('Component List: %s', cross_comp_max)
            '''
            steps += 1

    '''
    Tie Breaking Functions
    

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
            
    '''

    '''
    Deprecated: Weight DF Methods 
    

    def set_weight_df(self):
        index = [frozenset((u, v)) for u, v in itertools.combinations(self.dataset.keys(), 2)]
        self.weight_df = pd.DataFrame(index=index)
        self.weight_df['nb'] = self.nb_name
        self.weight_df['artifacts'] = len(self.dataset)
        # self.weight_df['rows'] = len(self.dataset['0.csv'].index)
        # self.weight_df['columns'] = len(self.dataset['0.csv'])
        return self.weight_df

    def check_edge_pair(self, key):
        if key in self.weight_df.index:
            return True
        else:
            self.logger.warning('Edge %s not present in Weight DF', key)
            return False

    def augment_ground_truth(self):
        self.weight_df['ground_truth'] = False
        # self.weight_df['operation'] = np.nan

        for u, v, data in self.g_truth.edges(data=True):
            key = frozenset((u, v))
            if self.check_edge_pair(key):
                self.weight_df.at[key, 'ground_truth'] = True
                self.weight_df.at[key, 'operation'] = data['operation']

        return self.weight_df

    def augment_inferred_graph(self):
        self.weight_df['g_inferred'] = False
        # self.weight_df['operation'] = np.nan

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
    
    '''

    def add_edges_and_merge_components(self, param, param1):
        pass

