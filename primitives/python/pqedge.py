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
        key, max_score = super().popitem()
        return_items.append((key, max_score))

        while super().topitem()[1] == max_score:
            return_items.append(super().popitem())

        return return_items