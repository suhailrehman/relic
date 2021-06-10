# Load a directory and return filename:df pairs
import logging
from collections import defaultdict
from itertools import chain

import pandas as pd
import glob
import os
import networkx as nx

import relic.distance.ppo
from relic.utils.pqedge import PQEdges


def build_df_dict(nb_name, base_dir):
    return build_df_dict_dir(base_dir + nb_name + '/artifacts/')


def build_df_dict_dir(csv_dir):
    dataset = {}
    for file in glob.glob(csv_dir + '*.csv'):
        csvfile = os.path.basename(file)
        try:
            dataset[csvfile] = pd.read_csv(file, index_col=0)
        except (pd.parser.CParserError, UnicodeDecodeError) as e:
            # Star Wars: encoding="ISO-8859-1"
            # df = pd.read_csv(
            # "http://math-info.hse.ru/f/2015-16/all-py/data/tariff2012.csv",
            # sep=';')
            if (csvfile == 'StarWars.csv'):
                dataset[csvfile] = pd.read_csv(file, encoding="ISO-8859-1", index_col=0)
            elif (csvfile == 'tariff2012.csv'):
                dataset[csvfile] = pd.read_csv(file, sep=";", index_col=0)
            else:
                print("Error reading file:", file)

    return dataset


def get_nb_dir(nb_file, base_dir):
    return base_dir + nb_file + '/'


def get_dataframe(nb_file, file, base_dir):
    artifact_dir = base_dir + nb_file + '/artifacts/'
    return pd.read_csv(artifact_dir + file, index_col=0)


def get_graph(nb_name, base_dir):
    result_file = base_dir + nb_name + '/' + nb_name + '_gt.pkl'
    return nx.read_gpickle(result_file)


def get_graph_edge_list(nb_name, metric, base_dir):
    result_file = base_dir + nb_name + '/inferred/infered_mst_' + metric + '.csv'
    # return nx.read_edgelist(result_file, delimiter=',', data=(('weight', float),))
    return nx.read_edgelist(result_file)


def get_distance_matrix(nb_name, metric, base_dir):
    result_file = base_dir + nb_name + '/inferred/' + metric + '_sim.csv'
    return pd.read_csv(result_file, index_col=0)


def check_csv_graph(artifact_dir, g_truth):
    missing_files = []
    for node in g_truth.nodes():
        if not os.path.exists(artifact_dir + node):
            print("Missing File: " + artifact_dir + node)
            missing_files.append(node)
    return missing_files


def combine_and_create_pkl(indir, outfile, ntuples=2):
    all_dfs = []
    for file in glob.glob(indir + '*.csv'):
        all_dfs.append(load_distances_from_file(file, ntuples=ntuples))

    pd.concat(all_dfs).sort_values('score', ascending=False).to_csv(outfile)


def load_distances_from_file(filename):
    score_df = pd.read_csv(filename, index_col=0)
    df_list = [x for x in score_df.columns if 'df' in x]
    scores_list = list(set(score_df.columns) - set(df_list))
    pairwise_scores = defaultdict(PQEdges)
    for label in scores_list:
        for ix, row in score_df.iterrows():
            k = [row[x] for x in df_list]
            key = ((k[0],k[1]), k[2]) if len(k) == 3 else frozenset(k)
            logging.debug(f'Dataframe variables: {k} : {key} : {row[label]}')
            pairwise_scores[label].additem(key, float(row[label]))

    return pairwise_scores


def _flatten_join(x):
    return x[0][0], x[0][1], x[1]


def _flatten_frozenset(x):
    return tuple(y for y in x)


def store_distances_to_file(pairwise_scores, filename, labels=None):
    if not labels:
        labels = [x for x in pairwise_scores.keys()]
    col_names = labels

    index = [x for x in pairwise_scores[labels[0]].keys()]
    score_df = pd.DataFrame(columns=col_names, index=[x for x in pairwise_scores[labels[0]].keys()])
    logging.debug(f'Loaded index: {index}')
    counter = 0
    for label in labels:
        logging.debug(f'Writing {label} values to DF')
        for k, v in pairwise_scores[label].items():
            logging.debug(f'Loading {k}:{v} to dataframe')
            score_df.at[k, label] = v
    if 'join' in labels:
        score_df.index = score_df.index.map(_flatten_join)
    else:
        score_df.index = score_df.index.map(_flatten_frozenset)

    logging.debug(f'Index Length: {score_df.index.nlevels}')
    logging.debug(f'Index : {score_df.index}')

    score_df.index = score_df.index.rename(['df'+str(x) for x in range(1, score_df.index.nlevels+1)])
    score_df.reset_index().to_csv(filename)


def store_all_distances(pairwise_scores, out_dir):
    logging.info(f'Writing distances to directory: {out_dir}')
    ppo_labels = []
    for label in pairwise_scores.keys():
        if label in relic.distance.ppo.PPO_LABELS:
            ppo_labels.append(label)
        else:
            store_distances_to_file({label: pairwise_scores[label]}, out_dir + '/'+label+'_scores.csv')

    if ppo_labels:
        store_distances_to_file({label: pairwise_scores[label] for label in ppo_labels},
                                out_dir+'/ppo_scores.csv')

    logging.info('Completed writing all distances to file')


def write_inferred_graph(g_inferred, output_file):
    nx.write_edgelist(g_inferred, output_file, data=True)