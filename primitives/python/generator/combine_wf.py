import sys
import glob
from shutil import copyfile
import networkx as nx
import os

_graphdir = 'graphs/'

def copy_artifact_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    os.makedirs(destination_dir+_graphdir, exist_ok=True)
    os.makedirs(destination_dir+'/artifacts/', exist_ok=True)

    nbs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    all_graphs = []
    for nb in nbs:
        artifact_dir = source_dir +'/' + nb + '/artifacts/'
        print('Copying:', artifact_dir)
        g_truth = nx.read_gpickle(source_dir+'/'+nb+'/'+nb+'_gt_fixed.pkl')
        g_truth_modified = nx.relabel_nodes(g_truth, lambda x: nb + '_' + x)
        # print(g_truth_modified.edges())
        all_graphs.append(g_truth_modified)
        nx.write_gpickle(g_truth_modified, destination_dir + _graphdir + nb + '_gt_fixed.pkl')
        for fpath in glob.glob(artifact_dir + '/*.csv'):
            fname = os.path.basename(fpath)
            new_fname = nb + '_' + fname
            copyfile(fpath, destination_dir + 'artifacts/' + new_fname)

    all_wf_g = nx.compose_all(all_graphs)
    wf_name = os.path.basename(destination_dir).strip('/')
    nx.write_gpickle(all_wf_g, destination_dir+wf_name+'_gt_fixed.pkl')


if __name__ == '__main__':
    copy_artifact_files(sys.argv[1], sys.argv[2])