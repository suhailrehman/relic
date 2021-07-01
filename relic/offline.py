from relic.distance.ppo import compute_all_ppo_labels, PPO_LABELS
from relic.distance.nppo import groupby_detector, pivot_detector, join_detector, check_join_schema
from relic.graphs.clustering import exact_schema_cluster
from relic.utils.serialize import build_df_dict_dir

import os
import itertools
import sys
import pandas as pd
import glob
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


_function_mappings = {
    'ppo': compute_all_ppo_labels,
    'groupby': groupby_detector,
    'pivot': pivot_detector,
    'join': join_detector
}


def enumerate_tuple_pairs(input_dir, out_filename, n_tuples=2, filter_function=None):
    # DO NOT LOAD tuples unless you need a schema filter_functions
    df_dict = None
    file_list = (os.path.basename(file) for file in glob.glob(input_dir + '*.csv'))
    if filter_function:
        # Limit join search to eligible clusters only
        df_dict = build_df_dict_dir(input_dir)
        file_list = df_dict.keys()

    i = 0
    with open(out_filename, 'w') as fp:
        for d in itertools.combinations(file_list, n_tuples):
            if filter_function:
                if not filter_function(*d, df_dict):
                    continue

            fp.write(f"{','.join(x for x in d)}\n")
            if i % 100000 == 0:
                logger.info(f'Written {i} records\r', )
            i += 1
    logger.info(f'Complete. Wrote  {i} records')


def enumerate_join_triples(cluster_dict=None, filename='join_combos.txt'):
    # Limit join search to eligible clusters only
    i = 0
    with open(filename, 'w') as fp:
        for c in itertools.combinations(cluster_dict.keys(), 3):
            if check_join_schema(c[0], c[1], c[2]):
                combos = [frozenset(i) for i in
                          itertools.product(cluster_dict[c[0]], cluster_dict[c[1]], cluster_dict[c[2]])]
                for d1, d2, d3 in combos:
                    # same_comp_count = sum([uf[d1] == uf[d2], uf[d2] == uf[d3], uf[d1] == uf[d3]])
                    # if same_comp_count <= 1:
                    fp.write(f"{d1},{d2},{d3}\n")
                    if i % 100000 == 0:
                        print(f'Written {i} records\r', )
                    i += 1


def compute_distance_pair(infile, out, input_dir, function=compute_all_ppo_labels, labels=PPO_LABELS):
    logger.info('Processing: ', infile, ' using ', function.__name__)
    file_part = os.path.basename(infile)
    df_dict = {}
    i = 0
    with open(out, 'w') as outfile:
        with open(infile, 'r') as infile:
            for line in infile:
                df_names = line.strip().split(',')


                # Load DF if not already in dict
                for dfn in df_names:
                    if dfn not in df_dict:
                        df_dict[dfn] = pd.read_csv(input_dir + dfn, index_col=0)

                dfs = [df_dict[dfn] for dfn in df_names]
                if function == join_detector:
                    score = (*df_names, df_dict)
                else:
                    score = function(*dfs, None)
                outfile.write(f"{','.join(x for x in df_names)},{score}\n")
                if i % 10000 == 0:
                    logger.info(f'{file_part}: Written {i} records\r', )
                i += 1

    logger.info(f'{file_part}: Complete. Wrote  {i} records')


def combine_and_create_pkl(in_dir, outfile):
    all_dfs = []
    for file in glob.glob(in_dir + '*.csv'):
        all_dfs.append(pd.read_csv(file, header=None, names=['df1', 'df2', 'score']))

    pd.concat(all_dfs).sort_values('score', ascending=False).to_csv(outfile)


def setup_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        help="Operation Mode: enumerate|compute|combine",
                        type=str, default='enumerate')

    parser.add_argument("--input",
                        help="Input path",
                        type=str, default='/tmp/')

    parser.add_argument("--slice",
                        help="Slice of work for compute process file",
                        type=str, default='/tmp/inferred/tuples/tuple_list_01.csv')

    parser.add_argument("--output",
                        help="Output path",
                        type=str, default='/tmp/inferred/')

    parser.add_argument("--func",
                        help="Distance Function",
                        type=str, default='ppo')

    # noinspection SpellCheckingInspection
    parser.add_argument("--ntuples",
                        help="Number of tuples to enumerate",
                        type=int, default=2)

    options = parser.parse_args(args)

    return options


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    options = setup_arguments(args)
    if options.mode == 'enumerate':
        if options.func == 'join':
            logger.info('Building Dataframe Dict...')
            df_dict = build_df_dict_dir(options.input)
            logger.info('Clustering by Schema...')
            cluster_dict = exact_schema_cluster(df_dict)
            logger.info('Enumerating Joins...')
            enumerate_join_triples(cluster_dict, options.output)
        else:
            enumerate_tuple_pairs(options.input, options.output, options.ntuples)
    elif options.mode == 'compute':
        compute_distance_pair(options.slice, options.output, options.input, function=_function_mappings[options.func])
    elif options.mode == 'combine':
        combine_and_create_pkl(options.input, options.output)


if __name__ == '__main__':
    main()
