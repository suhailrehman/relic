import itertools
from tqdm.autonotebook import tqdm

# Compute pairwise similarity metrics of dataset dict using similarity_metric
# returns reverse sorted list of (pair1, pair2, similarity_score) tuples
def get_pairwise_similarity_pc(dataset, distance_graph,  threshold=-1.0):
    pairwise_similarity = []
    pairs = list(itertools.combinations(dataset.keys(), 2))
    for d1, d2 in tqdm(pairs, desc='graph pairs', leave=False):
        score = distance_graph[d1][d2]['weight']
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            pass
            #print("WARNING: DROPPING",d1,d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity


def intra_cluster_similarity_pc(df_dict, clusters, distance_graph,  threshold=0.25):
    pairwise_jaccard = []
    for cluster in clusters.values():
        batch = {k: df_dict[k] for k in cluster}
        pw_batch = get_pairwise_similarity_pc(batch, distance_graph, threshold=threshold)
        pairwise_jaccard.extend(pw_batch)
    return pairwise_jaccard


def get_pairs_similarity_pc(dataset, cluster_set1, cluster_set2, distance_graph, threshold=-1.0):
    pairwise_similarity = []
    pairs = list(itertools.product(cluster_set1, cluster_set2))
    for d1, d2 in tqdm(pairs, desc='graph pairs', leave=False):
        if d1 == d2:
            continue
        score = distance_graph[d1][d2]['weight']
        if score >= threshold:
            pairwise_similarity.append((d1, d2, score))
        else:
            pass
            #print("WARNING: DROPPING",d1,d2, score, threshold)

    pairwise_similarity.sort(key=lambda x: x[2], reverse=True)
    return pairwise_similarity
