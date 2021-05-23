
import os
import itertools
import sys
import pandas as pd
import glob
import dataset as ds
import argparse

from lineage.similarity import compute_jaccard_DF, compute_colms_containment_DF, compute_col_jaccard_DF
from nppo import df_groupby_check_new, pivot_detector, score_join_schema, check_join_schema
from clustering import exact_schema_cluster

_function_mappings = {
    'celljaccard': compute_jaccard_DF,
    'cellcontainment': compute_colms_containment_DF,
    'columnjaccard': compute_col_jaccard_DF,
    'groupby': df_groupby_check_new,
    'pivot': pivot_detector,
    'join': score_join_schema
}

def enumerate_tuple_pairs(input_dir, outfilename, ntuples=2, filter_function=None):
    #DO NOT LOAD tuples unless you need a schema filter_functions
    df_dict=None
    filelist = (os.path.basename(file) for file in glob.glob(input_dir + '*.csv'))
    if filter_function:
        # Limit join search to eligible clusters only
        df_dict=ds.build_df_dict_dir(input_dir)
        filelist = df_dict.keys()

    i=0
    with open(outfilename, 'w') as fp:
        for d in itertools.combinations(filelist, ntuples):
            if filter_function:
                if not filter_function(*d,df_dict):
                    continue

            fp.write(f"{','.join(x for x in d)}\n")
            if i % 100000 == 0:
                print(f'Written {i} records\r',)
            i += 1
    print(f'Complete. Wrote  {i} records')


def enumerate_join_triples(cluster_dict=None, filename='joincombos.txt'):
    # Limit join search to eligible clusters only
    i=0
    with open(filename, 'w') as fp:
        for c in itertools.combinations(cluster_dict.keys(), 3):
            if check_join_schema(c[0], c[1], c[2]):
                combos = [frozenset(i) for i in itertools.product(cluster_dict[c[0]], cluster_dict[c[1]], cluster_dict[c[2]])]
                for d1,d2,d3 in combos:
                    #same_comp_count = sum([uf[d1] == uf[d2], uf[d2] == uf[d3], uf[d1] == uf[d3]])
                    #if same_comp_count <= 1:
                    fp.write(f"{d1},{d2},{d3}\n")
                    if i % 100000 == 0:
                        print(f'Written {i} records\r',)
                    i += 1


def compute_distance_pair(infile, out, input_dir, function=df_groupby_check_new):
    print('Processing: ', infile, ' using ', function.__name__)
    filepart = os.path.basename(infile)
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
                if function==score_join_schema:
                    score = (*df_names, df_dict)
                else:
                    score = function(*dfs, None)
                outfile.write(f"{','.join(x for x in df_names)},{score}\n")
                if i % 10000 == 0:
                    print(f'{filepart}: Written {i} records\r', )
                i += 1

    print(f'{filepart}: Complete. Wrote  {i} records')


def combine_and_create_pkl(indir, outfile):
    all_dfs = []
    for file in glob.glob(indir + '*.csv'):
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
                        type=str, default='celljaccard')

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
            print('Building Dataframe Dict...')
            df_dict = ds.build_df_dict_dir(options.input)
            print('Clustering by Schema...')
            cluster_dict = exact_schema_cluster(df_dict)
            print('Enumerating Joins...')
            enumerate_join_triples(cluster_dict,options.output)
        else:
            enumerate_tuple_pairs(options.input, options.output, options.ntuples)
    elif options.mode == 'compute':
        compute_distance_pair(options.slice, options.output, options.input, function=_function_mappings[options.func])
    elif options.mode == 'combine':
        combine_and_create_pkl(options.input, options.output)


if __name__ == '__main__':
    main()