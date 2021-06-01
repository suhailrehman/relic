import os.path
import glob
import pandas as pd


class ArtifactDict(dict):
    def __init__(self, artifact_dir):
        super().__init__()
        self.artifact_dir = artifact_dir

    def keys(self):
        return set(os.path.basename(file) for file in glob.glob(self.artifact_dir + '*.csv'))

    def items(self):
        for file in glob.glob(self.artifact_dir + '*.csv'):
            fname = os.path.basename(file)
            yield fname, pd.read_csv(self.artifact_dir+'/'+fname, index_col=0)

    def __missing__(self, key):
        self[key] = pd.read_csv(self.artifact_dir+'/'+key, index_col=0)
        return self[key]