import argparse
import copy
import json
import shutil
import sys
import zipfile
from datetime import datetime
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import UnionFind

from relic.distance.nppo import join_detector, groupby_detector, pivot_detector
from relic.distance.ppo import compute_all_ppo_labels
from relic.distance.tiebreakers import tiebreak_hash_edge, tiebreak_from_computed_scores
from relic.graphs.clustering import exact_schema_cluster, reverse_schema_dict, write_clusters_to_file
from relic.utils.pqedge import PQEdges, get_intra_cluster_edges_only
from relic.utils.serialize import build_df_dict_dir, write_graph, get_job_status_phases, update_phase, str2bool, \
    load_distances_from_raw_files
from relic.algorithm import compute_tuplewise_similarity

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


class RelicAlgorithm:

    def __init__(self, input_dir, output_dir, name='wf_', g_truth_file=None, **kwargs):
        logger.info('Starting instance of RelicAlgorithm on %s', name)

        # Directory Setup
        self.nb_name = name
        self.artifact_dir = input_dir
        self.inferred_dir = output_dir
        os.makedirs(self.inferred_dir, exist_ok=True)

        # Load the dataset
        # Load/read on demand infrastructure
        # self.dataset = ArtifactDict(self.artifact_dir)
        self.dataset = build_df_dict_dir(self.artifact_dir)

        # Create the initial graph
        self.g_inferred = nx.Graph()
        self.create_initial_graph()

        # Create initial components list and clustering
        self.components = UnionFind()
        self.num_components = 0
        self.initial_cluster = {}
        self.cluster_lookup = {}

        # Create the pairwise weights_dict to store multiple weights
        # Priority queue or self-sorting datastructure in association with unionfind
        self.pairwise_weights = defaultdict(PQEdges)
        self.initialize_components()

        # Load the Ground Truth
        # Optional GT annotation or remove entirely
        if g_truth_file:
            self.g_truth = nx.read_edgelist(g_truth_file)

        # Current Edge being added
        self.edge_no = 0

        # Tie Breaker Info
        self.tied_edges = {}
        self.two_tie_edges = {}

        # Serialization Info
        self.serialize = True
        self.score_records = dict()

    def load_artifacts(self):
        pass

    def create_initial_graph(self):
        logger.debug('Creating the initial graph of artifact nodes')
        self.g_inferred = nx.Graph()
        for artifact in self.dataset.keys():
            if artifact not in [n for n in self.g_inferred.nodes()]:
                logger.debug('Adding artifact to graph %s', str(artifact))
                self.g_inferred.add_node(artifact)

        return self.g_inferred

    def set_initial_clusters(self, cluster_type='exact_schema'):
        # Initial Clusters set to individual artifacts:
        if cluster_type == 'exact_schema':
            self.initial_cluster = exact_schema_cluster(self.dataset)
        self.cluster_lookup = reverse_schema_dict(self.initial_cluster)

        return self.initial_cluster

    def initialize_components(self):
        # Initializes the current list of graph components:
        logger.debug('Updating Graph Components')
        components = [n for c in nx.connected_components(self.g_inferred) for n in c]
        self.components = UnionFind(elements=components)
        self.num_components = len(components)
        return self.components

    def add_edge_and_merge_components(self, edge, data):
        logger.debug(f'About to add and merge edge {edge}')

        if type(edge[0]) == tuple:  # two sources, like join
            for e in edge[0]:
                self.g_inferred.add_edge(e, edge[1], **data)
                self.components.union(e, edge[1])
        else:
            self.g_inferred.add_edge(edge[0], edge[1], **data)
            self.components.union(edge[0], edge[1])

        logger.debug([x for x in self.components.to_sets()])
        self.edge_no += 1
        self.num_components = len([x for x in self.components])

    def compute_edges_of_type(self, edge_type='all', similarity_function=compute_all_ppo_labels,
                              n_pairs=2):
        edge_scores = compute_tuplewise_similarity(self.dataset, similarity_metric=similarity_function,
                                                                  label=edge_type, n_pairs=n_pairs)
        self.pairwise_weights.update(edge_scores)
        if self.serialize == True:
            for edge_type, score_dict in edge_scores.items():
                self.score_records[edge_type] = copy.deepcopy(score_dict)

    def add_edges_of_type(self, edge_type='jaccard', intra_cluster=False, tiebreak_function=None,
                          final_function=tiebreak_hash_edge,
                          sim_threshold=0.1, max_edges=None, max_step=False, tiebreak_kwargs=None):

        if not max_edges:
            max_edges = len(self.dataset)

        # Compute all pairwise edges first
        if edge_type not in self.pairwise_weights:
            raise KeyError('Edge type not computed in pairwise_weight dict')

        if intra_cluster:
            weights_pq = get_intra_cluster_edges_only(self.pairwise_weights[edge_type], self.cluster_lookup)
            logger.debug('Intra Cluster PQEdges')
            logger.debug(self.pairwise_weights[edge_type])
            logger.debug(weights_pq)
        else:
            weights_pq = self.pairwise_weights[edge_type]


        steps = 0
        stop = False
        if not max_step:
            max_step = len(self.dataset)

        while self.num_components > 1 and steps < max_step and not stop:
            logger.debug(f'PQ Status: {weights_pq}')
            logger.debug(f'UnionFind Status: {[x for x in self.components.to_sets()]}')
            max_edges = weights_pq.pop_unionfind_max(self.components)
            logger.debug(f'MaxEdges Found:  {max_edges} edge(s)...')
            if not max_edges:
                logger.debug('No compatible edges left in PQ/UnionFind')
                stop = True
                break
            elif max_edges[0][1] < sim_threshold:
                logger.debug(f'{max_edges} edge(s) below threshold {sim_threshold}, stopping')
                stop = True
                break
            elif len(max_edges) > 1:
                tie_break_edges = tiebreak_function(max_edges, **tiebreak_kwargs)
                if len(tie_break_edges) > 1:
                    edge = final_function(tie_break_edges)
                else:
                    edge = max_edges[0]
                for e in max_edges:
                    if e not in self.pairwise_weights[edge_type] and e != edge:
                        logger.debug(f'Adding back edge {e}, which is not {edge}')
                        try:
                            self.pairwise_weights[edge_type].additem(e[0], e[1])
                        except KeyError as e:
                            logger.warning(f'Trying to add back {e}, selected {edge}, to {self.pairwise_weights[edge_type].keys()}')
            else:
                edge = max_edges[0]

            weight = edge[1]

            logger.info('Adding edge #%d: %s, type %s', self.edge_no, list(edge), edge_type)
            self.add_edge_and_merge_components(list(edge[0]), {'weight': weight, 'type': edge_type, 'num': self.edge_no})
            steps += 1


def setup_arguments(args):
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--artifact_zip",
                        help="ZIP containing artifacts to be analyzed",
                        type=str)

    group.add_argument("--artifact_dir",
                        help="Directory containing artifacts to be analyzed",
                        type=str)

    parser.add_argument("--nb_name",
                        help="Name of the notebook to be analyzed",
                        type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))

    parser.add_argument("--out",
                        help="Output Directory for Results",
                        type=str, default='inferred/')

    parser.add_argument("--config_json",
                        help="Configuration JSON file for inference Job. Overrides all other job parameters if present",
                        type=str)

    parser.add_argument("--pre_cluster",
                        help="Pre cluster by exact schema",
                        type=str2bool, default=True,
                        nargs='?', const=True)

    parser.add_argument("--celljaccard",
                        help="Look for cell-level jaccard edges",
                        type=str2bool, default=True,
                        nargs='?', const=True)

    parser.add_argument("--cellcontain",
                        help="Look for cell-level containment edges",
                        type=str2bool, default=True,
                        nargs='?', const=True)

    parser.add_argument("--groupby",
                        help="Use Groupby Edge Detection",
                        type=str2bool, default=True,
                        nargs='?', const=True)

    parser.add_argument("--join",
                        help="Use Join Edge Detection",
                        type=str2bool, default=True,
                        nargs='?', const=True)

    parser.add_argument("--transform",
                        help="Use Transform Edge Detection",
                        type=str2bool, default=False,
                        nargs='?', const=False)

    parser.add_argument("--pivot",
                        help="Use Pivot Edge Detection",
                        type=str2bool, default=True,
                        nargs='?', const=True)

    parser.add_argument("--intra_cell",
                        help="Cell-Level Edge Retention Threshold",
                        type=float, default=0.1)

    parser.add_argument("--colt",
                        help="Column-Level Edge Retention Threshold",
                        type=float, default=0.1)

    parser.add_argument("--inter_cell",
                        help="Inter Cluster Cell-Level Edge Retention Threshold",
                        type=float, default=0.1)

    parser.add_argument("--g_truth_file",
                        help="Ground Truth File as a pickled NetworkX graph",
                        type=str)

    parser.add_argument("--pre_compute",
                        help="Use pre-computed distances (in the appropriate distance folder)",
                        type=str2bool, default=False,
                        nargs='?', const=False)

    options = parser.parse_args(args)

    return options


def run_relic(options):
    global logger
    logger.info('Testing RELIC on input:' + str(options.artifact_dir))
    logger.info('Output directory: ' + str(options.out))
    logger.info(f'Options: {options}')
    os.makedirs(options.out, exist_ok=True)

    # Setup Job Log
    logger = logging.getLogger()
    # create file handler which logs even debug messages
    fh = logging.FileHandler(str(options.out)+'job.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    job_status = {
        'status': 'running',
        'job_id': options.nb_name,
        'phaseno': 0,
        'totalphases': get_job_status_phases(options)
    }

    status_file = str(options.out)+'job_status.json'

    if 'artifact_zip' in options and options.artifact_zip: #Zip file was provided
        zip_out = options.out+'/artifacts/'
        logger.info(f'Extracting ZIP file: {options.artifact_zip} to directory {zip_out}')
        if os.path.exists(zip_out):
            shutil.rmtree(zip_out)
        os.makedirs(zip_out)
        with zipfile.ZipFile(options.artifact_zip, 'r') as zip_ref:
            zip_ref.extractall(zip_out)
        artifact_dir = zip_out
    else:
        artifact_dir = options.artifact_dir

    relic_instance = RelicAlgorithm(artifact_dir, options.out, name=options.nb_name, g_truth_file=options.g_truth_file)
    relic_instance.create_initial_graph()

    update_phase(job_status, 'Computing Pairwise Distances', status_file)
    if options.pre_compute:
        relic_instance.pairwise_weights.update(load_distances_from_raw_files(options.out+'/inferred/ppo/'))
    else:
        relic_instance.compute_edges_of_type(edge_type='all', similarity_function=compute_all_ppo_labels, n_pairs=2)

    if options.pre_cluster:
        update_phase(job_status, 'Clustering', status_file)
        clusters = relic_instance.set_initial_clusters()
        os.makedirs(options.out+'/inferred', exist_ok=True)
        write_clusters_to_file(clusters, options.out+'/inferred/clusters.txt')
        logger.info(f'Clustered by Schema: {len(clusters.keys())} individual clusters created.')
        logger.info('Looking for PPO edges within each cluster...')

        if options.celljaccard:
            update_phase(job_status, 'Intra-Cluster Jaccard', status_file)
            relic_instance.add_edges_of_type(edge_type='jaccard', intra_cluster=True,
                                             tiebreak_function=tiebreak_from_computed_scores,
                                             final_function=tiebreak_hash_edge,
                                             sim_threshold=options.intra_cell,
                                             tiebreak_kwargs={'pairwise_weights': relic_instance.pairwise_weights,
                                                              'score_type': 'overlap'}
                                             )

        if options.cellcontain:
            update_phase(job_status, 'Intra-Cluster Containment', status_file)
            relic_instance.add_edges_of_type(edge_type='containment', intra_cluster=True,
                                             tiebreak_function=tiebreak_from_computed_scores,
                                             final_function=tiebreak_hash_edge,
                                             sim_threshold=options.intra_cell,
                                             tiebreak_kwargs={'pairwise_weights': relic_instance.pairwise_weights,
                                                              'score_type': 'overlap'}
                                             )

    if options.join:
        logger.info('Looking for join edges across clusters...')
        update_phase(job_status, 'Join Detection', status_file)
        if options.pre_compute:
            relic_instance.pairwise_weights.update(load_distances_from_raw_files(options.out + '/inferred/join/'))
        else:
            relic_instance.compute_edges_of_type(edge_type='join', similarity_function=join_detector, n_pairs=3)
        relic_instance.add_edges_of_type(edge_type='join', intra_cluster=False, sim_threshold=1.0,
                                         tiebreak_function=tiebreak_from_computed_scores,
                                         final_function=tiebreak_hash_edge,
                                         tiebreak_kwargs={'pairwise_weights': relic_instance.pairwise_weights,
                                                          'score_type': 'overlap'}
                                         )

    logger.info('Looking for PPO edges across clusters...')
    if options.celljaccard:
        update_phase(job_status, 'Inter-Cluster Jaccard', status_file)
        relic_instance.add_edges_of_type(edge_type='jaccard', intra_cluster=False,
                                         tiebreak_function=tiebreak_from_computed_scores,
                                         final_function=tiebreak_hash_edge,
                                         sim_threshold=options.inter_cell,
                                         tiebreak_kwargs={'pairwise_weights': relic_instance.pairwise_weights,
                                                          'score_type': 'overlap'}
                                         )

    if options.cellcontain:
        update_phase(job_status, 'Inter-Cluster Containment', status_file)
        relic_instance.add_edges_of_type(edge_type='containment', intra_cluster=False,
                                         tiebreak_function=tiebreak_from_computed_scores,
                                         final_function=tiebreak_hash_edge,
                                         sim_threshold=options.inter_cell,
                                         tiebreak_kwargs={'pairwise_weights': relic_instance.pairwise_weights,
                                                          'score_type': 'overlap'}
                                         )
    if options.groupby:
        logger.info('Looking for Groupby edges across clusters...')
        update_phase(job_status, 'Groupby Detection', status_file)
        if options.pre_compute:
            relic_instance.pairwise_weights.update(load_distances_from_raw_files(options.out + '/inferred/groupby/'))
        else:
            relic_instance.compute_edges_of_type(edge_type='groupby', similarity_function=groupby_detector, n_pairs=2)
        relic_instance.add_edges_of_type(edge_type='groupby', intra_cluster=False, sim_threshold=1.0,
                                         tiebreak_function=tiebreak_from_computed_scores,
                                         final_function=tiebreak_hash_edge,
                                         tiebreak_kwargs={'pairwise_weights': relic_instance.pairwise_weights,
                                                          'score_type': 'containment'}
                                         )
    if options.pivot:
        logger.info('Looking for Pivot edges across clusters...')
        update_phase(job_status, 'Pivot Detection', status_file)
        if options.pre_compute:
            relic_instance.pairwise_weights.update(load_distances_from_raw_files(options.out + '/inferred/pivot/'))
        else:
            relic_instance.compute_edges_of_type(edge_type='pivot', similarity_function=pivot_detector, n_pairs=2)
        relic_instance.add_edges_of_type(edge_type='pivot', intra_cluster=False, sim_threshold=1.0,
                                         tiebreak_function=tiebreak_from_computed_scores,
                                         final_function=tiebreak_hash_edge,
                                         tiebreak_kwargs={'pairwise_weights': relic_instance.pairwise_weights,
                                                          'score_type': 'containment'}
                                         )

    write_graph(relic_instance.g_inferred, options.out + '/inferred_graph.csv')
    if options.g_truth_file:
        nx.write_edgelist(relic_instance.g_truth, options.out + '/true_graph.txt', data=True)

    job_status['status'] = 'complete'
    with open(status_file, 'w') as fp:
        json.dump(job_status, fp)


def main(args=sys.argv[1:]):
    logger.info(f'Arguments: {args}')
    options = setup_arguments(args)
    logger.info(f'Options: {options}')
    run_relic(options)


if __name__ == "__main__":
    main()
