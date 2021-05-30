from hashlib import md5


def hash_edge(x):
    w = "+".join(sorted(x[:-1])).encode('utf8')
    return md5(w).hexdigest()


def tiebreak_hash_edge(edges):
    return sorted(edges, key=hash_edge)


def hash_edge_join(x):
    w = "+".join((x[0][0], x[0][1], x[1])).encode('utf8')
    return md5(w).hexdigest()
