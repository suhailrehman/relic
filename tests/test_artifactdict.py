import glob
import os
import logging

import pytest
import numpy as np
import pandas as pd

from relic.utils.artifactdict import ArtifactDict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/test_workflow/artifacts/')
files = glob.glob(data_dir+'/*.csv')


@pytest.fixture
def artifactdict_instance():
    """Returns an ArtifactDict initialized to test_dir"""
    logging.info('Testing artifactdir on input: ' + str(data_dir))
    logging.debug('File List: ' + str(files))
    return ArtifactDict(data_dir)


def test_keys(artifactdict_instance):
    expected_files = set(os.path.basename(f) for f in files)
    actual_files = set(artifactdict_instance.keys())
    logging.debug('Keys: ' + str(expected_files))
    assert len(expected_files) != 0
    assert actual_files == expected_files


def test_missing(artifactdict_instance):
    test_file = np.random.choice([os.path.basename(f) for f in files], 1)[0]
    expected_columns = set(pd.read_csv(data_dir + test_file, index_col=0).columns)
    actual_columns = set(artifactdict_instance[test_file].columns)
    logging.debug('Expected Columns: ' + str(expected_columns))
    assert len(expected_columns) != 0
    assert actual_columns == expected_columns
