import os
import pytest
import logging
import glob
from itertools import product

from relic.core import RelicAlgorithm
from relic.utils.serialize import build_df_dict_dir
from relic.approx.sampling import load_df_sample, get_file_rowcount, generate_sample_index
from relic.core import setup_arguments, run_relic

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(THIS_DIR, 'data/test_workflow/artifacts/')
files = [os.path.basename(f) for f in glob.glob(data_dir+'*.csv')]
sample_sizes = [0.05, 0.1, 0.25, 0.5, 0.75]


@pytest.mark.parametrize("file, frac", product(files[:-1], sample_sizes))
def test_load_df_sample_random(file, frac):
    dataset = build_df_dict_dir(data_dir)
    normal_row_count = len(dataset[file].index)

    sampled_df = load_df_sample(data_dir+'/'+file, frac=frac)
    sample_row_count = len(sampled_df.index)
    actual_frac = sample_row_count/normal_row_count
    logging.debug(f"Frac: {frac, }Normal: {normal_row_count}, Sample: {sample_row_count}, Actual Frac: {actual_frac}")
    assert actual_frac == pytest.approx(frac, abs=0.1)


@pytest.mark.parametrize("frac", sample_sizes)
def test_load_df_sample_consistent(tmpdir_factory, frac):
    # dataset = build_df_dict_dir(data_dir)
    output_dir = tmpdir_factory.mktemp(f'inferred_sample_{frac}')
    sample_index = generate_sample_index(data_dir, output_dir, frac=frac)

    for file in files:
        original_df_size = get_file_rowcount(data_dir+'/'+file)
        sampled_df = load_df_sample(data_dir+'/'+file, frac=frac, sample_index=sample_index)
        #logging.info(f"sample_df: {sampled_df.head()}")
        if len(sample_index) > original_df_size:
            logging.debug(f'Testing File: {file}')
            logging.debug(f"original_index: {sorted(sample_index)}")
            logging.debug(f"df_sampled_index: {sorted(sampled_df.index)}")
            assert set(sampled_df.index).issubset(set(sample_index)) \
                , f"Difference: {set(sampled_df.index) - set(sample_index)}"


@pytest.mark.parametrize("frac, sample_index", product(sample_sizes, [False, True]))
def test_relic_online_sample_full(tmpdir_factory, frac, sample_index):
    output_dir = tmpdir_factory.mktemp('inferred_actual')
    logging.info('Testing Sampling RELIC on input:' + str(data_dir))
    logging.info('Temporary Output directory: '+str(output_dir))
    cmd_arguments = f"--artifact_dir={data_dir} --nb_name=test --out={output_dir} " + \
                    f"--sample_frac={frac} --sample_index={sample_index}"

    logging.info('Command Line: '+str(cmd_arguments))

    options = setup_arguments(cmd_arguments.split(" "))
    run_relic(options)
