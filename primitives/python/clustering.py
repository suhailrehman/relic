# Pre Clustering Functions

from collections import defaultdict
from tqdm.autonotebook import tqdm
import csv
import itertools
import networkx as nx
import numpy as np

from lineage import similarity, precomputed_sim



def exact_schema_cluster(df_dict):
    clusters = defaultdict(list)
    for fname, df in df_dict.items():
        clusters[frozenset(df)].append(fname)
    return clusters


def reverse_schema_dict(schema_dict):
    reverse_dict = {}
    for schema, artifact_list in schema_dict.items():
        for artifact in artifact_list:
            reverse_dict[artifact] = schema

    return reverse_dict


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


# Duplicate function
def set_jaccard_distance(set1, set2):
    intersect = set1.intersection(set2)
    union = set1.union(set2)
    return 1 - (len(intersect) / len(union))


def get_schema_distances(schemas):
    distance_dict = defaultdict(list)
    for combo in itertools.combinations(schemas.keys(), 2):
        distance_dict[set_jaccard_distance(*combo)].append(combo)

    return distance_dict


def get_merge_candidates(schema_distances, threhsold=0.8):
    min_distance = min(schema_distances.keys())

    if min_distance > threshold:
        return None, schema_distances

    # Select lowest distance pair to merge at this stage
    min_schema_pairs = schema_distances[min_distance]

    merge_schema = min_schema_pairs.pop(0)

    # Remove if the min_schema_pairs is empty:
    if len(min_schema_pairs) == 0:
        del (schema_distances[min_distance])

    return merge_schema, schema_distances


def generate_score_dict(pairscores):
    returndict = defaultdict(list)
    for src,dst,score in pairscores:
        returndict[score].append((src,dst,score))
    return returndict


def tiebreak_pairscores(df_dict, pairlist):
    max_pair = None
    max_raw_score = 0.0

    for src, dst, score in pairlist:
        srcdf = df_dict[src]
        dstdf = df_dict[dst]
        overlap_score = similarity.compute_DF_overlap(srcdf, dstdf)

        if overlap_score > max_raw_score:
            max_pair = (src, dst, score)
            max_raw_score = overlap_score

    return max_pair


def tiebreak_pairscores_col(df_dict, pairlist):
    min_pair = None
    min_raw_score = np.inf

    for src, dst, score in pairlist:
        srcdf = df_dict[src]
        dstdf = df_dict[dst]
        overlap_score = len(set(srcdf).symmetric_difference(set(dstdf)))

        if overlap_score < min_raw_score:
            min_pair = (src, dst, score)
            min_raw_score = overlap_score

    return min_pair


def find_components_join_edge(g_inferred, df_dict, pw_graph=None, threshold=0.01):
    schema_dict = exact_schema_cluster(df_dict)

    a_schema_dict = reverse_schema_dict(schema_dict)

    components = [c for c in nx.connected_components(g_inferred)]

    all_cmp_pairs_similarties = []

    for srccmp, dstcmp in itertools.combinations(components, 2):
        if pw_graph:
            similarites = precomputed_sim.get_pairs_similarity_pc(df_dict, srccmp, dstcmp, pw_graph)
        else:
            similarites = similarity.get_pairs_similarity(df_dict, srccmp, dstcmp)
        all_cmp_pairs_similarties.extend(similarites)

    all_cmp_pairs_similarties.sort(key=lambda x: x[2], reverse=True)

    score_dict = generate_score_dict(all_cmp_pairs_similarties)

    maxscore = max(score_dict)

    if maxscore > threshold:
        if len(score_dict[maxscore]) > 1:
            print("Breaking Tie for cell-level:", score_dict[maxscore])
            src, dst, score = tiebreak_pairscores(df_dict, score_dict[maxscore])
        else:
            src, dst, score = score_dict[maxscore][0]

        print('Adding edge', src, dst, score)
        g_inferred.add_edge(src, dst, weight=score)
        return g_inferred

    else:
        similarites = similarity.get_pairs_similarity(df_dict, srccmp, dstcmp, similarity_metric=similarity.compute_col_jaccard_DF)

        score_dict = generate_score_dict(similarites)

        maxscore = max(score_dict)

        if len(score_dict[maxscore]) > 1:
            print("Breaking Tie for column-level:", score_dict[maxscore])
            src, dst, score = tiebreak_pairscores_col(df_dict, score_dict[maxscore])
        else:
            src, dst, score = score_dict[maxscore][0]


        print('Adding column edge', src, dst, score)
        g_inferred.add_edge(src, dst, weight=score)
        return g_inferred
