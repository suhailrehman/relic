#!/usr/bin/env python
"""experiment.py: Code to handle primitive inputs/outputs"""

__author__ = "Suhail Rehman"

import dbconnection
import queries
import datetime
import uuid
import pandas as pd
import networkx as nx


DEFAULT_HASH = '12b53b3d96ffc567769f1fe49e745d85a0f47326'
REPO_URL = 'https://github.com/suhailrehman/relic/commit/'

''' EXPERIMENT FUNCTIONS '''


def start_new_experiment(parameters, commit_hash=DEFAULT_HASH):
    experiment_sql = '''INSERT INTO lineage.experiment(
    id, parameters, commit_hash) VALUES (%s, %s, %s, %s);'''

    conn = dbconnection.connect_db()
    experiment_id = uuid.uuid4()

    try:
        dbconnection.execute_query(experiment_sql,
                                   (str(experiment_id),
                                    str(parameters),
                                    commit_hash), conn)
    except Exception as e:
        raise e
        return None

    return experiment_id


def set_experiment_starttime(experiment_id, time=datetime.datetime.now()):
    experiment_sql = '''UPDATE lineage.experiment
    SET  start_time=(TIMESTAMP '%s') WHERE id=%s;'''

    conn = dbconnection.connect_db()
    try:
        dbconnection.execute_query(experiment_sql,
                                   (str(time.isoformat),
                                    str(experiment_id)), conn)
    except Exception as e:
        raise e
        return False

    return time


''' CLUSTER FUNCTIONS '''


def get_clusters_from_file(cluster_file):
    clusters = {}

    with open(cluster_file, 'r') as cf:
        content = cf.readlines()

    for line in content:
        line_list = line.strip().split(',')
        del line_list[1]
        clusters[line_list[0]] = line_list[1:]

    return clusters


def store_experiment_clusters(experiment_id, workflow_id, clusters,
                              cluster_type='exact_schema'):
    cluster_sql = '''INSERT INTO lineage.cluster(
    id, artifact_id, cluster_type, experiment_id, primitive_cluster_id)
    VALUES (%s, %s, %s, %s, %s);'''

    conn = dbconnection.connect_db()

    for primitive_cluster_id, artifacts in clusters.iteritems():
        cluster_id = uuid.uuid4()
        for artifact in artifacts:
            artifact_id = queries.get_artifact_id_name(artifact,
                                                       workflow_id, conn)
            try:
                dbconnection.execute_query(cluster_sql,
                                           (str(cluster_id),
                                            str(artifact_id),
                                            cluster_type,
                                            str(experiment_id),
                                            str(primitive_cluster_id)), conn)
            except Exception as e:
                raise e
                return False

    return True


''' EDGE WEIGHT FUNCTIONS '''


def get_graph_from_file(edge_weight_file):
    df = pd.read_csv(edge_weight_file, index_col=['name'])
    return nx.from_pandas_adjacency(df)


def store_relationship_edge(experiment_id, workflow_id, graph,
                            distance_type='cell_sim'):
    edge_sql = '''INSERT INTO lineage.relationship_edge(
    id, artifact_1, artifact_2, distance_type, distance_value,
    experiment_id) VALUES (%s, %s, %s, %s, %s, %s);'''

    conn = dbconnection.connect_db()

    artifact_ids = {}

    for a1, a2 in graph.edges:
        if a1 not in artifact_ids:
            artifact_ids[a1] = queries.get_artifact_id_name(a1,
                                                            workflow_id, conn)
        if a2 not in artifact_ids:
            artifact_ids[a2] = queries.get_artifact_id_name(a2,
                                                            workflow_id, conn)

        edge_id = uuid.uuid4()

        try:
            dbconnection.execute_query(edge_sql,
                                       (str(edge_id),
                                        str(artifact_ids[a1]),
                                        str(artifact_ids[a2]),
                                        distance_type,
                                        graph[a1][a2]['weight'],
                                        str(experiment_id)), conn)
        except Exception as e:
            raise e
            return False

    return True

#test_file='../primitives/cpp/src/pre_clustering/clusters_with_filename.csv'
test_file='../primitives/cpp/src/preserving_ops/result/cell_sim_2.csv'

g = get_graph_from_file(test_file)
for v1,v2 in g.edges:
    print(v1,v2,g[v1][v2]['weight'])
