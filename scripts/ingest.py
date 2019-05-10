#!/usr/bin/env python
"""ingest.py: Code to handle workflow/artifact ingestion"""

__author__ = "Suhail Rehman"

import uuid
import sys
import os
import glob
import argparse

import dbconnection


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


def store_ground_truth_nx(workflow, cursor):
    pass


'''
def main():
    """Main entry point for the script."""
    pass


if __name__ == '__main__':
    sys.exit(main())
'''


def test_workflow():
    directory = '/home/suhail/Scratch/pyexec/dataset/nb_123977.ipynb'
    return parse_directory_workflow(directory)


def test_store_workflow(workflow):
    conn = dbconnection.connect_db()
    return store_workflow(workflow, conn)


def test_store_artifacts(artifacts):
    conn = dbconnection.connect_db()
    return store_artifacts(artifacts, conn)


wf = test_workflow()
test_store_workflow(wf)
afs = wf['artifacts']
test_store_artifacts(afs)
