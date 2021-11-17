import numpy as np
import logging
from pytest import approx

from relic.approx.containment import sample_containment_estimator

_CONTAINMENT_EPSILON = 0.7


def test_sample_containment_estimator():
    query_set = set([1, 2, 3])
    source_set = set([1, 2, 3, 4, 5, 6, 7])
    source_sample_set = np.random.choice(list(source_set), 5, replace=False)
    source_set_len = len(source_set)
    sampling_ratio = len(source_sample_set) / source_set_len
    logging.info(f"Sample Source Set: {source_sample_set}")
    containment_estimate = sample_containment_estimator(query_set, source_sample_set, source_set_len,
                                                        sampling_ratio=sampling_ratio)
    true_containment_ratio = len(source_set.intersection(query_set)) / len(query_set)
    logging.info(f"True Containment: {true_containment_ratio} , estimate: {containment_estimate}")
    assert containment_estimate == approx(true_containment_ratio, abs=_CONTAINMENT_EPSILON, rel=_CONTAINMENT_EPSILON)

