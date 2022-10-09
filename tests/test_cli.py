import logging
import sys
import os
import pytest

from relic.core import main

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/test_workflow/artifacts/')
zip_file = os.path.join(THIS_DIR, 'data/test_workflow/artifacts.zip')


@pytest.mark.parametrize("src", [f"--artifact_dir={str(data_dir)}", f"--artifact_zip={str(zip_file)}"])
def test_main(tmpdir, src):
    output = tmpdir.mkdir("inferred_actual")
    logging.info('Testing RELIC on input:' + str(data_dir))
    logging.info('Temporary Output directory: ' + str(output))
    sys.argv = ['relic', f"{src}",
                f"--nb_name=test_workflow",
                f"--out={str(output)}"]
    main(args=sys.argv[1:])
    graph_file = output.join('/inferred_graph.csv')
    assert os.path.exists(str(graph_file))

