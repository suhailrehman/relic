import pqdict
from operator import lt, gt


class PQEdges(pqdict.pqdict):
    def __init__(self, max_q=True, *args, **kwargs):
        if max_q:
            super().__init__(*args, precedes=gt, **kwargs)
        else:
            super().__init__(*args, precedes=lt, **kwargs)

    def pop_max(self):
        """
            Return a list of all items in the pq dict with the same max score after popping them all
        """
        return_items = []
        try:
            key, max_score = super().popitem()
            return_items.append((key, max_score))
            while super().topitem()[1] == max_score:
                return_items.append(super().popitem())
        except KeyError as e:
            pass

        return return_items

    def pop_unionfind_max(self, components):
        return_items = []
        not_found = True
        try:
            while not_found:
                key, max_score = super().popitem()
                u, v = key

                if components[u] != components[v]:
                    return_items.append((key, max_score))
                    not_found = False

                while super().topitem()[1] == max_score:
                    key, score = super().popitem()
                    u, v = key
                    if components[u] != components[v]:
                        return_items.append((key, score))
                        not_found = False

        except KeyError as e:
            pass

        return return_items


def get_intra_cluster_edges_only(pq_edge, cluster_sets):
    new_pq = PQEdges()
    for edge in pq_edge.keys():
        u,v = edge
        if cluster_sets[u] == cluster_sets[v]:
            new_pq.additem(edge, pq_edge[edge])
    return new_pq



