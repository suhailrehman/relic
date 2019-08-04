import pandas as pd

# TODO: Rewrite as Class-based methods
def get_dataframe(csvdir, file):
    return pd.read_csv(csvdir+'/'+file, index_col=0)
