import os.path
import glob
import pandas as pd


class ArtifactDict(dict):
    def __init__(self, artifact_dir):
        super().__init__()
        self.artifact_dir = artifact_dir

    def keys(self):
        for file in glob.glob(self.artifact_dir + '*.csv'):
            yield os.path.basename(file)

    def __missing__(self, key):
        self[key] = pd.read_csv(self.artifact_dir+'/'+key, index_col=0)
        return self[key]