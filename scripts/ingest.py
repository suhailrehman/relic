#!/usr/bin/env python
"""ingest.py: Code to handle workflow/artifact ingestion"""

__author__ = "Suhail Rehman"

import uuid
import sys
import os
import glob
import argparse

import dbconnection
import queries

import networkx as nx


def setup_argparse():
    parser = argparse.ArgumentParser(description='Ingest Tools for RELIC.')
    parser.add_argument("-d", "--directory", type=str,
                        help="Workflow Directory path to ingest")
    parser.add_argument()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    return args


def get_file_details():
    pass


def parse_directory_workflow(directory):
    workflow_id = uuid.uuid4()
    artifacts = [{'filename': os.path.basename(path),
                  'id': uuid.uuid4(),
                  'workflow_id': workflow_id,
                  'path': path}
                 for path in glob.glob(directory+'/*.csv')]
    return {'id': workflow_id, 'directory': directory, 'artifacts': artifacts}


def store_artifacts(artifacts, connection):
    artifact_sql = '''INSERT INTO lineage.artifact(
    id, filename, path, workflow_id)
    VALUES (%s, %s, %s, %s);
    '''
    for artifact in artifacts:
        result = dbconnection.execute_insert(artifact_sql,
                                            (str(artifact['id']),
                                             artifact['filename'],
                                             artifact['path'],
                                             str(artifact['workflow_id'])),
                                            connection)
        if not result:
            print("ERROR inserting artifact: "+str(artifact))
            return False

    return True


def store_workflow(workflow, connection):
    workflow_sql = '''INSERT INTO lineage.workflow(id, directory_path)
    VALUES (%s, %s);
    '''
    return dbconnection.execute_insert(workflow_sql,
                                      (str(workflow['id']),
                                       workflow['directory']),
                                      connection)


def store_gt_edge(gt_edge, connection):
    gt_edge_sql = '''INSERT INTO lineage.ground_truth_edge(
    id, artifact_1, artifact_2, workflow_id)
    VALUES (%s, %s, %s, %s);
    '''

    return dbconnection.execute_insert(gt_edge_sql,
                                       (str(gt_edge['id']),
                                        gt_edge['artifact_1'],
                                        gt_edge['artifact_2'],
                                        gt_edge['workflow_id']),
                                       connection)


def load_graph(graph_pkl_file):
    return nx.read_gpickle(graph_pkl_file)


def store_ground_truth_nx(gt_workflow, wf_id, connection):
    try:
        for v1, v2 in gt_workflow.edges:
            af_id1 = queries.get_artifact_id_name(v1, wf_id, connection)
            af_id2 = queries.get_artifact_id_name(v2, wf_id, connection)
            # print(v1, af_id1, v2, af_id2)
            gt_edge = {
                'id': uuid.uuid4(),
                'artifact_1': af_id1,
                'artifact_2': af_id2,
                'workflow_id': wf_id
            }
            store_gt_edge(gt_edge, connection)
    except Exception as e:
        print(e)
        raise e
        return False

    return True


'''
def main():
    """Main entry point for the script."""
    pass


if __name__ == '__main__':
    sys.exit(main())
'''


def test_workflow():
    directory = '/home/suhail/Projects/sample_workflows/Full_Data/nb_123977/artifacts'
    return parse_directory_workflow(directory)


def test_store_workflow(workflow, conn):
    return store_workflow(workflow, conn)


def test_store_artifacts(artifacts, conn):
    return store_artifacts(artifacts, conn)

'''
conn = dbconnection.connect_db()
dbconnection.truncate_db(conn)
wf = test_workflow()
test_store_workflow(wf, conn)
afs = wf['artifacts']
test_store_artifacts(afs, conn)
path = '/home/suhail/Projects/sample_workflows/Full_Data/nb_123977/artifacts'
wf_id = queries.get_workflow_id_path(path, conn)
graph = load_graph('/home/suhail/Projects/sample_workflows/Full_Data/nb_123977/nb_123977_gt.pkl')
store_ground_truth_nx(graph, wf_id, conn)
'''

def ingest_workflow(wf_path, gt_graph_path, truncate=False):
    conn = dbconnection.connect_db()

    if truncate:
        dbconnection.truncate_db(conn)

    # Get workflow items and store them.
    workflow = parse_directory_workflow(wf_path)
    store_workflow(workflow, conn)

    # Store artifact_sql
    store_artifacts(workflow['artifacts'], conn)

    # Store Ground ground_truth
    graph = load_graph(gt_graph_path)
    store_ground_truth_nx(graph, str(workflow['id']), conn)

    return True


wf_path = '/home/suhail/Projects/sample_workflows/Full_Data/nb_123977/artifacts'
graph_path = '/home/suhail/Projects/sample_workflows/Full_Data/nb_123977/nb_123977_gt.pkl'

ingest_workflow(wf_path, graph_path, truncate=True)
