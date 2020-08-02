

class ColumnTypeException(Exception):
    "Raised when column type is not available for mutation"
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class TooSimilarException(Exception):
    "Raised when generated dataframe is too similar to one that's already generated"
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)