import os

import networkx as nx

from relic.core import *

import pytest
import logging

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/20210126-153738/artifacts/')
out_dir = os.path.join(THIS_DIR, 'data/20210126-153738/artifacts/')
file1 = '0.csv'
file2 = '1.csv'


@pytest.fixture
def relic_instance(tmpdir):
    """Returns relic initialized to test_dir and a random output dir"""
    d = tmpdir.mkdir("inferred")
    logging.info('Testing RELIC on input:' + str(data_dir))
    logging.info('Temporary Output directory: '+str(d))
    return RelicAlgorithm(data_dir, d, name='20210126-153738')


@pytest.mark.usefixtures("relic_instance")
class TestRelicAlgorithm:
    def test_create_initial_graph(self, relic_instance):
        logging.info('Testing initial graph creation')
        assert nx.is_empty(relic_instance.create_initial_graph())
