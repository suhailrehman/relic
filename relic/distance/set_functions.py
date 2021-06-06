def set_jaccard_distance(set1, set2):
    return 1 - set_jaccard_similarity(set1, set2)


def set_jaccard_similarity(set1, set2):
    intersect = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersect) / len(union)


def set_max_containment(set1, set2):
    intersect = len(set1.intersection(set2))
    if len(set1) < 1 or len(set2) < 1:
        return 0.0
    return max(intersect / len(set1), intersect / len(set2))


# Returns the containment of set1 in set2
def set_containment(set1, set2):
    intersect = len(set1.intersection(set2))
    return intersect / len(set1)
