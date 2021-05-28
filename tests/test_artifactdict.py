import glob
import os
import logging

import pytest

from relic.utils.artifactdict import ArtifactDict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, '../data/20210126-153738/artifacts/')
files = glob.glob(data_dir+'/*.csv')


@pytest.fixture
def artifactdict_instance():
    """Returns an ArtifactDict initialized to test_dir"""
    logging.info('Testing artifactdir on input:' + str(data_dir))
    return ArtifactDict(data_dir)


def test_keys(artifactdict_instance):
    actual_files = set(os.path.basename(f) for f in files)
    test_files = set(artifactdict_instance.keys())
    assert actual_files == test_files
