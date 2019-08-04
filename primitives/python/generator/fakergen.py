import pandas as pd
import numpy as np
import sys
import traceback
from faker import Faker
from lineage import LineageTracker
import networkx as nx
import argparse
import time
import os

class FakerVersionGenerator:

    def __init__(self, shape=(8,20), out_directory='dataset/',
                 num_base_versions=1, scale=10., gt_prefix='dataset'):

        self.functions = self.load_function_dict()
        self.inv_functions = self.inv_function_dict()

        self.fake = Faker()

        rowsize, colsize = shape

        self.dataset = []
        self.dataset_metadata = []
        base_df = self.generate_base_df(num_cols=colsize, num_rows=rowsize)
        self.out_directory = out_directory
        self.lineage = LineageTracker(self.out_directory+'artifacts/')
        self.dataset.append(base_df)
        self.lineage.new_item(self.get_last_label(), base_df)
        self.scale = scale
        self.gt_prefix = gt_prefix

    def load_function_dict(self, directory='./sources/'):
        return {
            'joinable': [line.rstrip('\n') for line in open(directory+'joinable_cols.txt')],
            'groupable': [line.rstrip('\n') for line in open(directory+'groupable_cols.txt')],
            'numeric': [line.rstrip('\n') for line in open(directory+'numeric_cols.txt')],
            'string': [line.rstrip('\n') for line in open(directory+'string_cols.txt')],
        }

    def inv_function_dict(self):
        return {v:k for k,vs in self.functions.items() for v in vs}

    def get_last_label(self):
        return str(len(self.dataset)-1)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def select_new_cols(self, group, num_cols, repeat=False):
        return np.random.choice(self.functions[group], num_cols, replace=repeat).tolist()

    def select_rand_cols(self, df, num=1):
        return np.random.choice(df.columns.values, num)

    def select_rand_col_group(self, df, group, num_cols):
        col_group = [col for col in df.columns.values
                     if col in self.inv_functions.keys()
                     and self.inv_functions[col] == group]

        #TODO: Fix case where num_cols is too large for available columns
        if col_group:
            return np.random.choice(col_group, num_cols).tolist()
        else:
            raise TypeError('Cannot select '+str(num_cols)+' number of columns of type '+group)

    def get_row_permutation(self, df):
        return np.random.permutation(df.index.values)

    def select_rand_dataset(self, for_merge=False):

        size = len(self.dataset)
        i = np.arange(size)  # an array of the index value for weighting
        prob = np.exp(i/self.scale)  # higher weights for larger index values
        prob /= prob.sum()

        if(for_merge):
            if(size < 2):
                return None
            elif(size == 2):
                return [0, 1]
            else:
                choice = np.random.choice(i, 2, p=prob,
                                          replace=False)
        else:
            if(size < 2):
                return 0
            choice = np.random.choice(i, 1, p=prob)[0]

        return choice

    def generate_base_df(self, num_cols=8, num_rows=20, atleast_one_pk=False,
                     repeat_cols=False, seed=None, index_col=None,
                     join_cols=None):
        #TODO: Cardinality Enforcement
        #TODO: Seed Selection
        #TODO: Index Column

        #TODO: Customizable Column Groups
        four_rands = np.random.multinomial(num_cols, np.ones(4)/4, size=1)[0]
        print('Selection Config: ', four_rands)
        num_joinable = four_rands[0]
        num_groupable = four_rands[1]
        num_numeric = four_rands[2]
        num_string = four_rands[3]

        selected_cols = []

        selected_cols.extend(self.select_new_cols('joinable', num_joinable))
        selected_cols.extend(self.select_new_cols('groupable', num_groupable))
        selected_cols.extend(self.select_new_cols('numeric', num_numeric))
        selected_cols.extend(self.select_new_cols('string', num_string))

        series = []

        print("Base DF: ", selected_cols)
        for col in selected_cols:
            col_type = self.inv_functions[col]
            if col_type == 'numeric':
                gen_col = pd.Series((self.fake.format(col) for _ in range(num_rows)), dtype='float64')
            else:
                gen_col = pd.Series((self.fake.format(col) for _ in range(num_rows)))
            series.append(gen_col)

        return pd.concat(series, axis=1, keys=selected_cols)

    def apply_op(self, op_function, **kwargs):
        if op_function == self.merge:  # Merge is special case

            #Select random dataset as left side of merge
            choice = self.select_rand_dataset()
            df1 = self.dataset[choice]

            #Select a join column
            #TODO: Handle tables without any join column to being with
            join_column = self.select_rand_col_group(df1, 'joinable', 1)[0]
            df2 = self.generate_base_df(num_rows=len(df1.index))

            #Add the join column to new df if not already present
            # TODO: Generate any length df2 and pad with newly generated faker values
            if join_column not in df2.columns.values:
                df2[join_column] = df1[join_column].values

            # Append newly generated merge table as dataset item:
            merge_tbl_ver = str(len(self.dataset))
            self.lineage.new_item(merge_tbl_ver, df2)
            self.dataset.append(df2)

            # Perform the merge and save result as new version
            new_df = self.merge(df1, df2, on=join_column).dropna()
            self.lineage.new_item(str(len(self.dataset)), new_df)
            self.dataset.append(new_df)
            self.lineage.link(str(choice), self.get_last_label(),
                              str(op_function.__name__))
            self.lineage.link(merge_tbl_ver, self.get_last_label(),
                              str(op_function.__name__))

        else:
            choice = self.select_rand_dataset()
            base_df = self.dataset[choice]
            new_df = op_function(base_df, **kwargs)
            if new_df.empty:
                raise pd.errors.EmptyDataError
            new_df = new_df.dropna()
            self.lineage.new_item(str(len(self.dataset)), new_df)
            self.dataset.append(new_df)
            self.lineage.link(str(choice), self.get_last_label(),
                              str(op_function.__name__))

    def select_rand_aggregate(self):
        return np.random.choice(['min', 'max', 'sum', 'mean', 'count'], 1)[0]

    def assign(self, df):
        # TODO: Select string or numeric
        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        # TODO: Randomize assignment function
        new_col_name = str(col)+"_pow3"
        print("Selected for apply:", col)
        # Dynamic keyword expansion in python below:
        return df.assign(**{new_col_name: lambda x: np.power(x[col], 3)}) #np.exp(x[col])

    def groupby(self, df):
        #TODO: Ensure groupable columns exist in dataframe
        col = self.select_rand_col_group(df, 'groupable', 1)
        func = self.select_rand_aggregate()
        print("Grouping By: ", col[0], 'aggregation: ', func)
        method = getattr(df.groupby(col[0]), func)
        new_df = method().dropna(axis=1) #TODO: Verify drop behavior
        return method()

    def iloc(self, df):
        # Select random row slice
        num1 = np.random.randint(0, len(df.index))
        num2 = np.random.randint(num1, len(df.index))
        return df.iloc[num1:num2]

    def nlargest(self, df):
        n = np.random.randint(len(df.index)/2, max(2,len(df.index)))
        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        if col:
            return df.nlargest(n, col)
        else:
            return None

    def nsmallest(self, df):
        n = np.random.randint(len(df.index)/2, max(2,len(df.index)))
        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        if col:
            return df.nsmallest(n, col)
        else:
            return None

    def reindex(self, df):
        return df.reindex(self.get_row_permutation(df))

    def get_rand_percentage(minimum=0.01, maximum=0.99):
        return round(np.random.random_sample(), 2)

    def sample(self, df):
        return df.sample(frac=self.get_rand_percentage())

    def sort_values(self, df):
        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        choice = np.random.choice([True, False], 1)[0]
        if col:
            return df.sort_values(by=col, ascending=choice)
        else:
            return None

    def merge(self, df1, df2, on=None):
        return df1.merge(df2, on=on)

    def select_rand_op(self):
        operations = [
            #self.agg,       ### non-preserving
            self.assign,
            #self.expanding,
            self.groupby,   ### non-preserving
            self.iloc,
            #self.melt,
            self.merge,     ### non-preserving
            self.nlargest,
            self.nsmallest,
            #self.reindex,   ### this may cause a problem??
            #self.rolling,
            self.sample,
            #self.sort_index,
            self.sort_values,
            #self.pivot   #TODO
        ]

        return np.random.choice(operations, 1)[0]

    def write_graph_files(self):
        def csv_mapping(x):
            return x+'.csv'

        csv_graph = nx.relabel_nodes(self.lineage.graph, csv_mapping)

        nx.write_gpickle(csv_graph, self.out_directory+self.gt_prefix+'_gt.pkl')
        nx.write_edgelist(csv_graph, self.out_directory+self.gt_prefix+'_gt_edgelist.txt')


def generate_dataset(shape, n, output_dir, scale=10., gt_prefix='dataset'):

    dataset = FakerVersionGenerator(shape, scale=scale, out_directory=output_dir, gt_prefix=gt_prefix)

    errors = []

    i = 0

    while i < n-1:
        choice = dataset.select_rand_op()
        try:
            print("Version: "+str(i)+" applying: "+ str(choice.__name__))
            dataset.apply_op(choice)
            i += 1
        except pd.errors.EmptyDataError as e:
            print("Empty DF result")
            pass
        except TypeError as e:
            print("Cannot apply operation because of missing column type")
            pass
        except Exception as e:
            tb = traceback.format_exc()
            errors.append({choice: tb})
            raise

    dataset.write_graph_files()
    return dataset, errors


def setup_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir",
                        help="Location of Output datasets to be stored",
                        type=str, default='dataset')

    parser.add_argument("--prefix",
                        help="prefix for each workflow to be generated\
                        dir to be the path prefix for these files.",
                        type=str)

    parser.add_argument("--col",
                        help="Number of columns in the base version",
                        type=int, default=6)

    parser.add_argument("--row",
                        help="Number of rows in the base version",
                        type=int, default=100)

    parser.add_argument("--ver",
                        help="Number of versions to generate",
                        type=int, default=10)

    parser.add_argument("--bfactor",
                        help="Workflow Branching factor, 0.1 is linear, 100 is star-like",
                        type=float, default=10.0)


    options = parser.parse_args(args)

    return options


def main(args=sys.argv[1:]):

    #TODO: store csv files in output_dir + 'artifacts'
    options = setup_arguments(args)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    out_dir = options.output_dir+'/'+timestr+'/'
    os.makedirs(out_dir, exist_ok=True)
    csv_dir = out_dir+'artifacts/'
    os.makedirs(csv_dir, exist_ok=True)


    generate_dataset((options.row,options.col),options.ver,out_dir,scale=options.bfactor, gt_prefix=timestr)


if __name__ == '__main__':
    main()
