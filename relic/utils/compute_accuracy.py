import networkx as nx
import sys
import json

from relic.graphs.graphs import get_precision_recall

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

relic_result=nx.read_edgelist(sys.argv[1])
ground_truth=nx.read_gpickle(sys.argv[2])

print(json.dumps(get_precision_recall(ground_truth, relic_result), cls=SetEncoder, indent=2))

