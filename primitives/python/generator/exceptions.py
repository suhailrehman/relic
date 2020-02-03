

class ColumnTypeException(Exception):
    "Raised when column type is not available for mutation"
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)