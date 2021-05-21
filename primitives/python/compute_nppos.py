
import nppo
import os
import itertools
import sys
import pandas as pd
import glob
import dataset as ds
import clustering

def enumerate_nppo_pairs(input_dir, outfilename):
    df_dict=ds.build_df_dict_dir(input_dir)
    # Limit join search to eligible clusters only
    i=0
    with open(outfilename, 'w') as fp:
        for d1, d2 in itertools.combinations(df_dict.keys(), 2):
            fp.write(f"{d1},{d2}\n")
            if i % 100000 == 0:
                print(f'Written {i} records\r',)
            i += 1
    print(f'Complete. Wrote  {i} records')


def run_groupby_score(infile, outfile, input_dir):
    print('Processing: ', infile)
    filepart = os.path.basename(infile)
    df_dict = {}
    i = 0
    with open(outfile, 'w') as outfile:
        with open(sys.argv[1], 'r') as infile:
            for line in infile:
                df_names = line.strip().split(',')

                # Load DF if not already in dict
                for dfn in df_names:
                    if dfn not in df_dict:
                        df_dict[dfn] = pd.read_csv(input_dir + dfn, index_col=0)

                score = nppo.df_groupby_check_new(*df_names, df_dict, None)
                outfile.write(f"{df_names[0]},{df_names[1]},{score}\n")
                if i % 10000 == 0:
                    print(f'{filepart}: Written {i} records\r', )
                i += 1

    print(f'{filepart}: Complete. Wrote  {i} records')

def combine_and_create_pkl(indir, outfile):
    all_dfs = []
    for file in glob.glob(indir + '*.csv'):
        all_dfs.append(pd.read_csv(file, header=None, names=['df1', 'df2', 'score']))

    pd.concat(all_dfs).sort_values('score', ascending=False).to_csv(outfile)

if __name__ == '__main__':
    #enumerate_nppo_pairs(sys.argv[1], sys.argv[2])
    run_groupby_score(sys.argv[1], sys.argv[2], sys.argv[3])
    #combine_and_create_pkl(sys.argv[1], sys.argv[2])