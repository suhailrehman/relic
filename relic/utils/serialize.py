# Load a directory and return filename:df pairs
import pandas as pd
import glob
import os
import networkx as nx


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


def load_distances_from_file(filename, ntuples=2):
    col_names = ['df'+str(x) for x in range(1, ntuples+1)].append('score')
    return pd.read_csv(filename, header=None, names=col_names)
