
import nppo
import os
import itertools
import sys
import pandas as pd
import glob
import pickle
import dataset as ds
import clustering

def enumerate_join_triples(cluster_dict=None, filename='joincombos.txt'):
    # Limit join search to eligible clusters only
    i=0
    with open(filename, 'w') as fp:
        for c in itertools.combinations(cluster_dict.keys(), 3):
            if nppo.check_join_schema(c[0], c[1], c[2]):
                combos = [frozenset(i) for i in itertools.product(cluster_dict[c[0]], cluster_dict[c[1]], cluster_dict[c[2]])]
                for d1,d2,d3 in combos:
                    #same_comp_count = sum([uf[d1] == uf[d2], uf[d2] == uf[d3], uf[d1] == uf[d3]])
                    #if same_comp_count <= 1:
                    fp.write(f"{d1},{d2},{d3}\n")
                    if i % 100000 == 0:
                        print(f'Written {i} records\r',)
                    i += 1


input_dir = '/tank/local/suhail/data/relic/mixed/syn2/artifacts/'
#output_filename = '/tank/local/suhail/data/relic/mixed/syn2/inferred/join_triples.txt'
#df_dict = ds.build_df_dict_dir(input_dir)
#cluster_dict = clustering.exact_schema_cluster(df_dict)
#enumerate_join_triples(cluster_dict=cluster_dict, filename=output_filename)

def run_join_score(infile, outfile):
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

                score = nppo.score_join_schema(*df_names, df_dict)
                outfile.write(f"{score[0][0]},{score[0][1]},{score[1]},{score[2]}\n")
                if i % 10000 == 0:
                    print(f'{filepart}: Written {i} records\r', )
                i += 1

    print(f'{filepart}: Complete. Wrote  {i} records')

def combine_and_create_pkl(indir, outfile):
    all_dfs = []
    for file in glob.glob(indir + '*.txt'):
        all_dfs.append(pd.read_csv(file, header=None, names=['df1', 'df2', 'df3', 'score']))

    all_join_vals = pd.concat(all_dfs, ignore_index=True) #.sort_values('score', ascending=False).to_csv(outfile)

    triple_dict = {}

    for ix, row in all_join_vals.iterrows():
        #print(row['df1'])
        k = frozenset((row['df1'], row['df2'], row['df3']))
        v = ((row['df1'], row['df2']), row['df3'], row['score'])
        triple_dict[k] = v
        #print(k,v)
        if ix % 10000 == 0:
            print(f'Loaded {ix} records to dict')


    with open(outfile, 'wb') as fp:
        pickle.dump(triple_dict, fp)

if __name__ == '__main__':
    #run_join_score(sys.argv[1], sys.argv[2])
    combine_and_create_pkl(sys.argv[1], sys.argv[2])