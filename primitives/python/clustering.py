# Pre Clustering Functions

from collections import defaultdict, OrderedDict
from tqdm.auto import tqdm
import csv
import itertools
import networkx as nx
import numpy as np

from lineage import similarity, precomputed_sim

from hashlib import md5


def hash_edge(x):
    w = "+".join(sorted(x[:-1])).encode('utf8')
    return md5(w).hexdigest()


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


def generate_tiebreak_score_dict(pairscores):
    returndict = defaultdict(list)
    for src,dst,score,tiebreak_score in pairscores:
        returndict[tiebreak_score].append((src,dst,score))
    return returndict


def tiebreak_pairscores(df_dict, pairlist):
    max_pair = None
    max_raw_score = 0.0

    scores_list = []

    for src, dst, score in pairlist:
        srcdf = df_dict[src]
        dstdf = df_dict[dst]
        overlap_score = similarity.compute_DF_overlap(srcdf, dstdf)

        scores_list.append((src,dst,score, overlap_score))

        if overlap_score >= max_raw_score:
            max_pair = (src, dst, score)
            max_raw_score = overlap_score

    score_dict = generate_tiebreak_score_dict(scores_list)

    if len(score_dict[max_raw_score])>1:
        print("Multiple Overlap candidates: ", score_dict[max_raw_score])
        s_edge_list = sorted(score_dict[max_raw_score], key=hash_edge)
        return s_edge_list[0]


    return max_pair


def tiebreak_pairscores_col(df_dict, pairlist):
    min_pair = None
    min_raw_score = np.inf

    scores_list = []

    for src, dst, score in pairlist:
        srcdf = df_dict[src]
        dstdf = df_dict[dst]
        overlap_score = len(set(srcdf).symmetric_difference(set(dstdf)))

        scores_list.append((src,dst,score, overlap_score))

        if overlap_score <= min_raw_score:
            min_pair = (src, dst, score)
            min_raw_score = overlap_score


    score_dict = generate_tiebreak_score_dict(scores_list)

    if len(score_dict[min_raw_score])>1:
        print("Multiple Min SymDiff candidates.", score_dict[min_raw_score])
        s_edge_list = sorted(score_dict[min_raw_score], key=hash_edge)
        return s_edge_list[0]

    return min_pair


def tiebreak_pairscores_cell(df_dict, pairlist):
    max_pair = None
    max_cell_score = 0.0

    scores_list = []

    for src, dst, score in pairlist:
        srcdf = df_dict[src]
        dstdf = df_dict[dst]
        cell_score = similarity.compute_jaccard_DF(srcdf, dstdf)

        scores_list.append((src,dst,score,cell_score))

        if cell_score >= max_cell_score:
            max_pair = (src, dst, score)
            max_cell_score = cell_score

    score_dict = generate_tiebreak_score_dict(scores_list)

    if len(score_dict[max_cell_score])>1:
        print("Multiple Cell-Level candidates.", score_dict[max_cell_score])
        s_edge_list = sorted(score_dict[max_cell_score], key=hash_edge)
        return s_edge_list[0]


    return max_pair


def tiebreak_pairscores_minsize(df_dict, pairlist):
    # Get the smallest of max pair sizes to use for cell containment scoring.
    min_pair = None
    min_raw_score = np.inf

    scores_list = []

    for src, dst, score in pairlist:
        srcdf = df_dict[src]
        dstdf = df_dict[dst]
        overlap_score = max(srcdf.size, dstdf.size)

        scores_list.append((src,dst,score, overlap_score))

        if overlap_score <= min_raw_score:
            min_pair = (src, dst, score)
            min_raw_score = overlap_score


    score_dict = generate_tiebreak_score_dict(scores_list)

    if len(score_dict[min_raw_score])>1:
        print("Multiple Min Size candidates.", score_dict[min_raw_score])
        s_edge_list = sorted(score_dict[min_raw_score], key=hash_edge)
        return s_edge_list[0]

    return min_pair


def find_components_join_edge(g_inferred, df_dict, edge_num, pw_graph=None, col_pw_graph=None, cell_threshold=0.01, col_threshold=0.01, col=False,
                              cell_label='cell', col_label='col', primary_tie_break_function=tiebreak_pairscores, secondary_tie_break_function=tiebreak_pairscores_cell):
    schema_dict = exact_schema_cluster(df_dict)

    a_schema_dict = reverse_schema_dict(schema_dict)

    components = [c for c in nx.connected_components(g_inferred)]

    all_cmp_pairs_similarties = []
    added_edge = None

    src = None
    dst = None

    for srccmp, dstcmp in itertools.combinations(components, 2):
        if pw_graph:
            similarites = precomputed_sim.get_pairs_similarity_pc(df_dict, srccmp, dstcmp, pw_graph)
        else:
            similarites = similarity.get_pairs_similarity(df_dict, srccmp, dstcmp)
        all_cmp_pairs_similarties.extend(similarites)

    all_cmp_pairs_similarties.sort(key=lambda x: x[2], reverse=True)

    considered_edges = [frozenset((u,v)) for u,v,s in all_cmp_pairs_similarties]

    score_dict = generate_score_dict(all_cmp_pairs_similarties)

    maxscore = max(score_dict)

    if maxscore > cell_threshold:
        if len(score_dict[maxscore]) > 1:
            print("Breaking Tie for primary-level:", score_dict[maxscore])
            src, dst, score = primary_tie_break_function(df_dict, score_dict[maxscore])
        else:
            src, dst, score = score_dict[maxscore][0]

        print('Adding primary edge', src, dst, score)
        g_inferred.add_edge(src, dst, weight=score, num=edge_num, type=cell_label)
        edge_num += 1
        return g_inferred, edge_num, considered_edges, (src,dst)

    elif col:
        all_cmp_pairs_similarties = []

        for srccmp, dstcmp in itertools.combinations(components, 2):
            if col_pw_graph:
                similarites = precomputed_sim.get_pairs_similarity_pc(df_dict, srccmp, dstcmp, col_pw_graph)
            else:
                similarites = similarity.get_pairs_similarity(df_dict, srccmp, dstcmp, similarity_metric=similarity.compute_col_jaccard_DF)
            all_cmp_pairs_similarties.extend(similarites)

        all_cmp_pairs_similarties.sort(key=lambda x: x[2], reverse=True)

        score_dict = generate_score_dict(all_cmp_pairs_similarties)

        maxscore = max(score_dict)

        if maxscore > col_threshold:
            if len(score_dict[maxscore]) > 1:
                print("Breaking Tie for column-level using secondary tie breaker:", score_dict[maxscore])
                src, dst, score = secondary_tie_break_function(df_dict, score_dict[maxscore])
            else:
                src, dst, score = score_dict[maxscore][0]


            print('Adding secondary edge', src, dst, score)
            g_inferred.add_edge(src, dst, weight=score, num=edge_num, type=col_label)
            edge_num +=1

    else:
        print("No more edges above threshold")
        return None, edge_num, considered_edges, None

    if not src or not dst:
        return g_inferred, edge_num, considered_edges, None

    return g_inferred, edge_num, considered_edges, (src, dst)


def find_components_col_edge(g_inferred, df_dict, edge_num, col_pw_graph=None, col_threshold=0.01):
    schema_dict = exact_schema_cluster(df_dict)

    a_schema_dict = reverse_schema_dict(schema_dict)

    components = [c for c in nx.connected_components(g_inferred)]

    all_cmp_pairs_similarties = []

    src = None
    dst = None

    for srccmp, dstcmp in itertools.combinations(components, 2):
        if col_pw_graph:
            similarites = precomputed_sim.get_pairs_similarity_pc(df_dict, srccmp, dstcmp, col_pw_graph)
        else:
            similarites = similarity.get_pairs_similarity(df_dict, srccmp, dstcmp, similarity_metric=similarity.compute_col_jaccard_DF)
        all_cmp_pairs_similarties.extend(similarites)

    all_cmp_pairs_similarties.sort(key=lambda x: x[2], reverse=True)
    considered_edges = [frozenset((u, v)) for u, v, s in all_cmp_pairs_similarties]

    score_dict = generate_score_dict(all_cmp_pairs_similarties)

    maxscore = max(score_dict)

    if maxscore > col_threshold:
        if len(score_dict[maxscore]) > 1:
            print("Breaking Tie for column-level:", score_dict[maxscore])
            src, dst, score = tiebreak_pairscores_cell(df_dict, score_dict[maxscore])
        else:
            src, dst, score = score_dict[maxscore][0]


        print('Adding inter-cluster column edge', src, dst, score)
        g_inferred.add_edge(src, dst, weight=score, num=edge_num, type='col')
        edge_num +=1

    else:
        print("No more edges above threshold")
        return None, edge_num, considered_edges, None

    if not src or not dst:
        return g_inferred, edge_num, considered_edges, None

    return g_inferred, edge_num, considered_edges, (src, dst)




def max_spanning_tree(pw_graph, edge_type='cell'):
    G = nx.Graph()
    i = 0
    for i, e in enumerate(nx.maximum_spanning_edges(pw_graph)):
        G.add_edge(e[0],e[1], weight=pw_graph[e[0]][e[1]]['weight'], num=i, type=edge_type)

    return G, i+1



def max_spanning_tree_tie_breaker(pw_graph, g_truth=None, edge_type='cell', tiebreaker=None, df_dict=None):
    G = nx.Graph()
    i = 0
    for i, e in enumerate(max_spanning_edges_tie_breaker(pw_graph, g_truth, tiebreaker=tiebreaker, df_dict=df_dict)):
        print('Adding edge number', i, ':', e, 'weight:', pw_graph[e[0]][e[1]]['weight'])
        G.add_edge(e[0],e[1], weight=pw_graph[e[0]][e[1]]['weight'], num=i, type=edge_type)

    return G, i+1


def tiebreak_spanning_edges(edge_list, df_dict):
    # Get the smallest of max pair sizes to use for cell containment scoring.
    scores_list = defaultdict(list)

    for src, dst, data in edge_list:
        srcdf = df_dict[src]
        dstdf = df_dict[dst]
        overlap_score = max(srcdf.size, dstdf.size)

        scores_list[overlap_score].append((src,dst, data))

    return scores_list[min(scores_list.keys())]



def max_spanning_edges_tie_breaker(G, g_truth=None, weight='weight', data=True, tiebreaker=None, df_dict=None):
    from networkx.utils import UnionFind
    if G.is_directed():
        raise nx.NetworkXError("Mimimum spanning tree not defined for directed graphs.")

    subtrees = UnionFind()
    edges_dict = OrderedDict()

    for e in sorted(G.edges(data=True), key=lambda t: t[2].get(weight, 1), reverse=True):
        edges_dict.setdefault(e[2][weight],[]).append(e)

    g_truth_copy = None

    if g_truth:
        g_truth_copy = g_truth.to_undirected()

    for w, edge_list in edges_dict.items():
        #print('Column Weight:', w)
        if len(edge_list) > 1:
            edge_list_1 = edge_list
            if tiebreaker:
                print('TieBreaking with min/max containment')
                print(edge_list)
                edge_list_1 = tiebreaker(edge_list, df_dict)

            print('Number of Edges with weight:', w, len(edge_list_1))
            s_edge_list = sorted(edge_list_1, key=hash_edge)
            print('Sorted & Hash Values:', s_edge_list, list(map(hash_edge, s_edge_list)))
            for edge in s_edge_list:
                if g_truth_copy:
                    if not g_truth_copy.has_edge(edge[0],edge[1]):
                        continue
                if subtrees[edge[0]] != subtrees[edge[1]]:
                        print('Selecting Tie Breaker edge', edge, edge[0], edge[1])
                        yield (edge[0], edge[1], edge[2])
                        subtrees.union(edge[0], edge[1])

        for edge in edge_list:
            if subtrees[edge[0]] != subtrees[edge[1]]:
                yield (edge[0], edge[1], edge[2])
                subtrees.union(edge[0], edge[1])


