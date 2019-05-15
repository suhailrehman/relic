#!/usr/bin/env python
"""query.py: Code to handle query postgres db for specific items"""

__author__ = "Suhail Rehman"

import dbconnection
import experiment


def get_workflow_id_path(directory_path, connection):
    query = '''SELECT id FROM lineage.workflow WHERE workflow.directory_path LIKE %s'''
    r = dbconnection.execute_query(query, (directory_path,), connection)
    if r:
        return r[0][0]
    else:
        return None


def get_artifact_id_path(directory_path, connection):
    query = '''SELECT id FROM lineage.artifact WHERE artifact.path LIKE %s'''
    r = dbconnection.execute_query(query, (directory_path,), connection)
    if r:
        return r[0][0]
    else:
        return None


def get_artifact_id_name(filename, workflow_id, connection):
    query = '''SELECT id FROM lineage.artifact WHERE artifact.filename LIKE %s AND artifact.workflow_id == %s'''
    r = dbconnection.execute_query(query, (filename, workflow_id), connection)
    if r:
        return r[0][0]
    else:
        return None


def test_artifact_id_path():
    connection = dbconnection.connect_db()
    path = "/home/suhail/Scratch/pyexec/dataset/nb_123977.ipynb/user.csv"
    id = get_artifact_id_path(path, connection)
    return id


def get_code_base_experiment(experiment_id, connection):
    query = '''SELECT commit_hash FROM lineage.experiment WHERE experiment.id
    LIKE %s '''
    r = dbconnection.execute_query(query, (experiment_id), connection)
    if r:
        return experiment.REPO_URL+r[0][0]
    else:
        return None

def get_edge_by_artifacts(experiment_id, a1, a2, distance_type='cell_sim', connection):
    edge_query = '''SELECT id, artifact_1, artifact_2, distance_value FROM lineage.relationship_edge
    AS e WHERE e.artifact_1 = %s AND e.artifact_2 = %s AND e.experiment_id =
    %s AND e.distance_type=%s;'''

    r = dbconnection.execute_query(edge_query, (str(a1),
                                                str(a2),
                                                experiment_id,
                                                distance_type), connection)
    if r:
        return r[0]
    else:
        return None
