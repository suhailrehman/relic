# Pre Clustering Functions

from collections import defaultdict
import csv

def exact_schema_cluster(df_dict):
    clusters = defaultdict(list)
    for fname, df in df_dict.items():
        clusters[frozenset(df)].append(fname)
    return clusters

def write_clusters_to_file(clusters, cluster_file):
    with open(cluster_file, 'w') as fp:
        for i, cluster in enumerate(clusters.values()):
            fp.write("%d,%d,%s\n" % (i, len(cluster), ",".join(cluster)))


def get_graph_clusters(clusters_file):
    with open(clusters_file, 'r') as fp:
        reader = csv.reader(fp)
        cluster_list = list(reader)
        cluster_assign = {}
        for cluster in cluster_list:
            cluster_id = cluster[0]
            items = cluster[2:]
            for item in items:
                cluster_assign[item] = int(cluster_id)
        return cluster_assign
