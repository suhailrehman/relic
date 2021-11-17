

def sample_containment_estimator(query_set, source_sample_set, sampling_ratio=0.5):
    """
    Returns the adjusted containment estimation of the queryset in the sample source set
    for a given sampling ratio.
    Args:
        source_set_len: true size of the sampled source set of values
        query_set: set of values whose containment needs to be estimated
        source_sample_set: source set of values to estimate the containment of query against
        sampling_ratio: The sampling ratio used to generate teh source sample set (default 0.5)

    Returns:
        Containment ratio of the query set in the source set adjusted
        via the sampling ratio
    """
    common_values = len(query_set.intersection(source_sample_set))
    containment_estimate = min(((1.0 / sampling_ratio) * (common_values)) / len(query_set), 1.0)
    return containment_estimate


def sample_col_containment(df1, df2, df1_sample, colname, sampling_ratio, col2name=None):
    if col2name is None:
        col2name = colname

    df1valset = set(df1_sample[colname])
    df2valset = set(df2[col2name])


    return sample_containment_estimator(df2valset, df1valset, sampling_ratio=sampling_ratio)

