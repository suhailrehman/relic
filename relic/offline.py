import os
import itertools
import sys
from collections import defaultdict

import networkx as nx
import pandas as pd
import glob
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic.utils.analysis import is_join_op
from relic.distance.ppo import compute_all_ppo_labels, PPO_LABELS, compute_baseline_labels
from relic.distance.nppo import groupby_detector, pivot_detector, join_detector, check_join_schema, \
    sample_groupby_detector
from relic.graphs.clustering import exact_schema_cluster
from relic.utils.serialize import build_df_dict_dir, str2bool
from relic.approx.sampling import load_df_sample, generate_sample_index

import logging
logging.basicConfig(format='%(asctime)s %(name)s:%(lineno)d %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_function_mappings = {
    'ppo': compute_all_ppo_labels,
    'groupby': groupby_detector,
    'pivot': pivot_detector,
    'join': join_detector,
    'baseline' : compute_baseline_labels,
    'sample_groupby': sample_groupby_detector
}


def enumerate_tuple_pairs(input_dir, out_filename, n_tuples=2, filter_function=None):
    # DO NOT LOAD tuples unless you need a schema filter_functions
    df_dict = None
    logger.debug(f'Checking input dir: {input_dir}')
    file_list = (os.path.basename(file) for file in glob.glob(input_dir + '/*.csv'))
    #logger.debug(f'File List : {file_list}')
    if filter_function:
        # Limit join search to eligible clusters only
        df_dict = build_df_dict_dir(input_dir)
        file_list = df_dict.keys()

    i = 0
    with open(out_filename, 'w') as fp:
        #logger.debug(len([x for x in itertools.combinations(file_list, n_tuples)]))
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
                        logger.info(f'Written {i} records\r')
                    i += 1


def enumerate_gt_op_tuples(gt_graph_file='gt_fixed.pkl', op_type='join', filename='gt_join_combos.txt'):
    # Enumerate using ground truth
    gt_graph = nx.read_gpickle(gt_graph_file)
    nb_op_tuples = set()
    for u,v, uv_data in gt_graph.edges(data=True):
        if op_type == 'all':
            nb_op_tuples.add(frozenset([u, v]))
        elif op_type == 'join' and is_join_op(uv_data):
            # Create join triple from this edge\
            for s, t, st_data in gt_graph.in_edges(v, data=True):
                if is_join_op(st_data) and s != u:
                    nb_op_tuples.add(frozenset([u, s, v]))
        elif uv_data['operation'] == op_type:
            nb_op_tuples.add(frozenset([u, v]))

    i = 0
    with open(filename, 'w') as fp:
        for tup in nb_op_tuples:
            fp.write(f"{','.join(list(tup))}\n")
            i += 1

    logger.info(f'Complete. Wrote  {i} records')


def compute_distance_pair(infile, out, input_dir, function=compute_all_ppo_labels, frac=None, sample_index=None):
    logger.info(f'Processing: {infile} using  {function.__name__}')
    file_part = os.path.basename(infile)
    df_dict = {}
    i = 0
    if function == compute_all_ppo_labels:
        labels = PPO_LABELS
    elif 'detector' in function.__name__:
        labels = [function.__name__.split('_')[0]]
    elif 'baseline' in function.__name__:
        labels = ['baseline']

    ntuples = 3 if function == join_detector else 2
    with open(out, 'w') as outfile:
        # write header
        header=','.join(['df'+str(x) for x in range(1,ntuples+1)] + [x for x in labels])+'\n'
        outfile.write(header)
        with open(infile, 'r') as infile:
            for line in infile:
                df_names = line.strip().split(',')
                # Load DF if not already in dict
                if function == sample_groupby_detector:
                    sampled_df_dict = {}
                    df1_name, df2_name = df_names
                    if df1_name not in sampled_df_dict:
                        sampled_df_dict[df1_name] = load_df_sample(input_dir + df1_name, frac=frac,
                                                                   sample_index=sample_index)
                        sampled_df_dict[df2_name] = load_df_sample(input_dir + df2_name, frac=frac,
                                                                   sample_index=sample_index)
                    if df1_name not in df_dict:
                        df_dict[df1_name] = pd.read_csv(input_dir + df1_name, index_col=0)
                    if df2_name not in df_dict:
                        df_dict[df2_name] = pd.read_csv(input_dir + df2_name, index_col=0)
                    edge_tuple, scores = function(df1_name, df2_name, df_dict, sampled_df_dict, frac)
                else:
                    for dfn in df_names:
                        if dfn not in df_dict:
                            if frac:
                                df_dict[dfn] = load_df_sample(input_dir + dfn, frac=frac, sample_index=sample_index)
                            else:
                                df_dict[dfn] = pd.read_csv(input_dir + dfn, index_col=0)
                    dfs = [df_dict[dfn] for dfn in df_names]
                    edge_tuple, scores = function(*df_names, df_dict)

                if function == join_detector:  # Explicit edge ordering for join detector
                    df_names = edge_tuple[0][0], edge_tuple[0][1], edge_tuple[1]
                scores_list_str = ','.join([str(scores[l]) for l in scores.keys()])
                outfile.write(f"{','.join(x for x in df_names)},{scores_list_str}\n")
                logger.debug(f'{scores}')
                if i % 10000 == 0:
                    logger.info(f'{file_part}: Written {i} records\r', )
                i += 1

    logger.info(f'{file_part}: Complete. Wrote  {i} records')


def combine_and_create_pkl(in_dir, outfile):
    all_dfs = []
    for file in glob.glob(in_dir + '/*.csv'):
        all_dfs.append(pd.read_csv(file))

    pd.concat(all_dfs).to_csv(outfile)


def setup_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        help="Operation Mode: enumerate|gt_enumerate|compute|combine",
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

    parser.add_argument("--sample_frac",
                        help="Sampling fraction to read from each artifact",
                        type=float)

    parser.add_argument("--sample_index",
                        help="Consistent Sample Index File",
                        type=str, default=None)

    parser.add_argument("--gt_graph_file",
                        help="Ground Truth Graph File",
                        type=str, default=None)

    options = parser.parse_args(args)

    return options


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    options = setup_arguments(args)
    logger.debug(f'Options: {options}')
    if options.mode == 'sample':
        generate_sample_index(options.input, options.output, frac=options.sample_frac)
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
    elif options.mode == 'gt_enumerate':
        enumerate_gt_op_tuples(gt_graph_file=options.gt_graph_file,
                               op_type=options.func, filename=options.output)
    elif options.mode == 'compute':
        if options.sample_index:
            sample_index = []
            with open(options.sample_index) as fp:
                for line in fp.readlines():
                    sample_index.append(int(line.strip()))
        else:
            sample_index = None
        compute_distance_pair(options.slice, options.output, options.input,
                              function=_function_mappings[options.func],
                              sample_index=sample_index,
                              frac=options.sample_frac)
    elif options.mode == 'combine':
        combine_and_create_pkl(options.input, options.output)


if __name__ == '__main__':
    main()
