import dbconnection


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


def test_artifact_id_path():
    connection = dbconnection.connect_db()
    path = "/home/suhail/Scratch/pyexec/dataset/nb_123977.ipynb/user.csv"
    id = get_artifact_id_path(path, connection)
    return id
