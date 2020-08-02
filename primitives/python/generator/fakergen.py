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
from exceptions import *

from tqdm.auto import tqdm



def compute_jaccard_DF(df1,df2, pk_col_name=None):

    # fill NaN values in df1, df2 to some token val
    df1 = df1.fillna('jac_tmp_NA')
    df2 = df2.fillna('jac_tmp_NA')

    try:
        if(pk_col_name):
            df3 = df1.merge(df2, how='outer', on=pk_col_name, suffixes=['_jac_tmp_1','_jac_tmp_2'])
        else:
            df3 = df1.merge(df2, how='outer', left_index=True, right_index=True, suffixes=['_jac_tmp_1','_jac_tmp_2'])
    except TypeError as e:
        # print("Can't Merge")
        return 0

    # Get set of column column names:
    comparison_cols = set(col for col in df3.columns if'_jac_tmp_' in str(col))
    common_cols = set(col.split('_jac_tmp_',1)[0] for col in comparison_cols)

    if(len(common_cols) == 0):
        return 0

    # Get set of non-common columns:
    uniq_cols = set(col for col in df3.columns if'_jac_tmp_' not in str(col))
    if(pk_col_name):
        uniq_cols.remove(pk_col_name)

    # Check common cols and print True/False
    for col in common_cols:
        left = col+'_jac_tmp_1'
        right = col+'_jac_tmp_2'
        df3[col] = df3[left] == df3[right]

    # Unique columns are already false

    for col in uniq_cols:
        df3[col] = False

    #Drop superflous columns
    df3 = df3.drop(columns=comparison_cols)
    if(pk_col_name):
        df3 = df3.drop(columns=[pk_col_name])

    # Compute Jaccard Similarity
    intersection = np.sum(np.sum(df3))
    union = df3.size
    #print(intersection, union)

    return float(intersection) / union


class FakerVersionGenerator:

    def __init__(self, shape=(8,20), out_directory='dataset/',
                 num_base_versions=1, scale=10., gt_prefix='dataset', npp=False, matfreq=1):

        self.functions = self.load_function_dict()
        self.inv_functions = self.inv_function_dict()

        self.fake = Faker()

        rowsize, colsize = shape

        self.dataset = []
        self.dataset_metadata = []
        base_df = self.generate_base_df(num_cols=colsize, num_rows=rowsize)
        self.out_directory = out_directory
        os.makedirs(self.out_directory+'artifacts/', exist_ok=True)
        self.lineage = LineageTracker(self.out_directory+'artifacts/')
        self.dataset.append(base_df)
        self.lineage.new_item(self.get_last_label(), base_df)
        self.scale = scale
        self.gt_prefix = gt_prefix
        self.npp = npp
        self.matfreq = matfreq
        self.opcount = 0
        self.currentdf = None
        self.lastmatchoice = 0

        self.lastargs = None
        self.op_equv_set = []

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

        #Fixed in Selection: Fix case where num_cols is too large for available columns
        if col_group and len(col_group) >= num_cols:
            return np.random.choice(col_group, num_cols, replace=False).tolist()
        else:
            raise ColumnTypeException('Cannot select '+str(num_cols)+' number of columns of type '+group)

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

    def get_column_counts(self, num_cols):
        four_rands = np.random.multinomial(num_cols, np.ones(4)/4, size=1)[0]
        difference = 0
        for i, category in enumerate(self.functions.keys()):
            if four_rands[i] > len(self.functions[category]):
                difference += four_rands[i] - len(self.functions[category])
                four_rands[i] = len(self.functions[category])

        for i, category in enumerate(self.functions.keys()):
            to_add = min(difference, len(self.functions[category])-four_rands[i])
            four_rands[i] += to_add
            difference -= to_add

        return four_rands



    def generate_base_df(self, num_cols=8, num_rows=20, exclusions=None, atleast_one_pk=False,
                     repeat_cols=False, seed=None, index_col=None,
                     join_cols=None):
        #TODO: Cardinality Enforcement
        #TODO: Seed Selection
        #TODO: Index Column (Rely on Autonumbered Index for now)

        #TODO: Customizable Column Groups
        four_rands = self.get_column_counts(num_cols)
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

        if exclusions:
            for e in exclusions:
                if e in selected_cols:
                    selected_cols.remove(e)

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
            print('Join Column: ', join_column)
            duplicate_columns = set(df1.columns)
            df2 = self.generate_base_df(num_rows=len(df1.index), exclusions=duplicate_columns)

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

            if type(new_df) is not pd.DataFrame or new_df.empty:
                raise pd.errors.EmptyDataError

            self.lineage.new_item(str(len(self.dataset)), new_df)
            self.dataset.append(new_df)
            self.lineage.link(str(choice), self.get_last_label(),
                              str(op_function.__name__))
            self.lineage.link(merge_tbl_ver, self.get_last_label(),
                              str(op_function.__name__))

        else:
            if self.opcount == 0:
                choice = self.select_rand_dataset()
                self.lastmatchoice = choice
                base_df = self.dataset[choice]
            else:
                base_df = self.currentdf

            new_df = op_function(base_df, **kwargs)

            if type(new_df) is not pd.DataFrame or new_df.empty:
                raise pd.errors.EmptyDataError

            # Check if newly generated DF is identical (cell-wise to previous):
            if compute_jaccard_DF(new_df, base_df) == 1.0:
                raise TooSimilarException

            for d in tqdm(self.dataset):
                if compute_jaccard_DF(d, new_df) == 1.0:
                    raise TooSimilarException


            #new_df = new_df.dropna()
            self.currentdf = new_df
            self.opcount +=1

            if self.opcount == self.matfreq:
                self.lineage.new_item(str(len(self.dataset)), new_df)
                self.dataset.append(new_df)
                self.lineage.link(str(self.lastmatchoice), self.get_last_label(),
                                  str(op_function.__name__))
                self.opcount = 0
                self.currentdf = None

    def select_rand_aggregate(self):
        return np.random.choice(['min', 'max', 'sum', 'mean', 'count'], 1)[0]

    def assign_numeric(self, df):
        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        random_scalar = np.random.choice(range(2, 20), 1)[0]
        random_func = np.random.choice(['pow', 'exp', 'log', 'cumsum'], 1)[0]

        new_col_name = col+'__'+random_func+str(random_scalar)

        if new_col_name in df.columns:
            print("Assigned Column already exists")
            return None

        print("Applying function", random_func, "to column", col)

        if random_func == 'pow':
            return df.assign(**{new_col_name: lambda x: np.power(x[col], random_scalar)})
        elif random_func == 'exp':
            return df.assign(**{new_col_name: lambda x: np.exp(x[col])})
        elif random_func == 'log':
            return df.assign(**{new_col_name: lambda x: np.log(x[col])})
        else:
            return df.assign(**{new_col_name: lambda x: np.nancumsum(x[col].astype('float64'))})

    def assign_string(self, df):
        col = self.select_rand_col_group(df, 'string', 1)[0]
        new_col_name = col+'__swapcase'

        if new_col_name in df.columns:
            print("Assigned Column already exists")
            return None

        try:
            if type(df[col].iloc[0]) == list:
                return df.assign(**{new_col_name: lambda x: x[col].apply(lambda y: ",".join(y).swapcase())})
            else:
                return df.assign(**{new_col_name: lambda x: x[col].astype(str).apply(lambda y: y.swapcase())})
        except IndexError as e:
            print(df, col)
            return None

    def assign(self, df):
        types = ['string', 'numeric']
        string_or_numeric = np.random.choice(['string', 'numeric'], 1)[0]

        if string_or_numeric == 'string':
            return self.assign_string(df)
        else:
            return self.assign_numeric(df)


    def groupby(self, df):
        #TODO: Ensure groupable columns exist in dataframe
        col = self.select_rand_col_group(df, 'groupable', 1)
        #try:
        #    self.select_rand_col_group(df, 'numeric', 1)[0] # Raises error if no numerics
        #    func = self.select_rand_aggregate()
        #except ColumnTypeException as e:
        func = 'count'
        print("Grouping By: ", col[0], 'aggregation: ', func)
        method = getattr(df.groupby(col[0]), func)
        new_df = method().reset_index() #TODO: Verify drop behavior
        if len(new_df.index) == len(df.index):
            return None # No Contraction: GroupBy doesnt make sense here.
        return new_df

    def iloc(self, df):
        # Select random row slice
        # Maybe select at-least 10% of the rows?
        if len(df.index) < 10:
            return None
        stop = False
        while not stop:
            num1 = np.random.randint(0, len(df.index))
            num2 = np.random.randint(num1, len(df.index))
            print("Random iloc range %:", (num2 - num1) / len(df.index))
            if (num2 - num1) / len(df.index) >= 0.1:
                stop = True

        return df.iloc[num1:num2]

    def nlargest(self, df):
        stop = False
        while not stop:
            n = np.random.randint(len(df.index)/2, max(2,len(df.index)))
            if n / len(df.index) >= 0.1:
                stop = True

        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        if col:
            return df.nlargest(n, col)
        else:
            return None

    def nsmallest(self, df):
        stop = False
        while not stop:
            n = np.random.randint(len(df.index) / 2, max(2, len(df.index)))
            if n / len(df.index) >= 0.1:
                stop = True

        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        if col:
            return df.nsmallest(n, col)
        else:
            return None

    def reindex(self, df):
        return df.reindex(self.get_row_permutation(df))

    def get_rand_percentage(self, minimum=0.1, maximum=0.99):
        return round((maximum - minimum) * np.random.random_sample() + minimum,  2)

    def sample(self, df):
        return df.sample(frac=self.get_rand_percentage())

    def sort_values(self, df):
        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        choice = np.random.choice([True, False], 1)[0]
        if col:
            return df.sort_values(by=col, ascending=choice)
        else:
            return None

    def point_edit(self, df):
        col = self.select_rand_cols(df, 1)[0]
        print("Selected Column", col)
        if col:
            colvalues = set(df[col].values)
            if not colvalues:
                return None
            old_value = np.random.choice(list(colvalues), 1)[0]
            colname = col.split('__')[0]
            if colname not in self.inv_functions:
                return None
            if self.inv_functions[colname] == 'numeric':
                new_value = float(self.fake.format(colname))
            else:
                new_value = self.fake.format(colname)
            print("Split", colname)
            while new_value == old_value: #In case we end up with same value
                old_value = np.random.choice(list(colvalues), 1)[0]
                new_value = self.fake.format(colname)
            print("Replacing", col,"value",old_value,"with", new_value)
            new_df = df.copy()
            new_df.loc[new_df[col] == old_value, col] = new_value
            return new_df

        else:
            return None

    def dropcol(self, df):
        col = self.select_rand_cols(df, 1)[0]
        if col:
            print("Dropping column", col)
            new_df = df.copy()
            new_df = df.drop(col, axis=1)
            return new_df

        else:
            return None

    def merge(self, df1, df2, on=None):
        return df1.merge(df2, on=on, suffixes=['__x','__y'])

    def pivot(self, df):
        index, column = self.select_rand_col_group(df, 'groupable', 2)
        numeric = self.select_rand_col_group(df, 'numeric', 1)[0]
        print('Pivoting using index:', index, ' column:', column, 'and values:', numeric)

        newdf = df.pivot_table(index=index, columns=column, values=numeric, aggfunc=sum)
        newdf.columns = newdf.columns.map(str)
        newdf.rename(columns = {x: str(x)+'__pivoted' for x in newdf.columns})

        return newdf




    def select_rand_op(self):
        operations = [
            #self.agg,       ### non-preserving
            self.assign,
            #self.expanding,
            #self.iloc,
            #self.melt,
            #self.nlargest,
            #self.nsmallest,
            #self.reindex,   ### this may cause a problem??
            #self.rolling,
            self.sample,
            #self.sort_index,
            #self.sort_values,
            self.point_edit,
            self.dropcol
        ]

        if self.npp:
            operations.extend(
                [
                    self.merge,
                    self.groupby,
                    self.pivot
                ]
            )


        operations = {
            self.assign: 0.122,
            self.sample: 0.066,
            self.dropcol: 0.066,
            self.point_edit: 0.096,
            self.groupby: 0.333,
            self.merge: 0.276,
            self.pivot: 0.041
        }

        ops, probs = zip(*operations.items())

        return np.random.choice(ops, 1, p=probs)[0]

    def write_graph_files(self):
        def csv_mapping(x):
            return x+'.csv'

        csv_graph = nx.relabel_nodes(self.lineage.graph, csv_mapping)

        nx.write_gpickle(csv_graph, self.out_directory+self.gt_prefix+'_gt_fixed.pkl')
        nx.write_edgelist(csv_graph, self.out_directory+self.gt_prefix+'_gt_edgelist.txt')







def generate_dataset(shape, n, output_dir, scale=10., gt_prefix='dataset', npp=False, matfreq=1):

    dataset = FakerVersionGenerator(shape, scale=scale, out_directory=output_dir, gt_prefix=gt_prefix, npp=npp, matfreq=matfreq)

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
        except ColumnTypeException as e:
            print(e)
            print("Cannot apply operation because of missing column type, skipping")
            pass
        except TooSimilarException as e:
            print(e)
            print("Cannot apply operation because generated dataframe is too similar to ones already generated, skipping")
            pass

        except Exception as e:
            print(dataset.lastmatchoice)
            tb = traceback.format_exc()
            errors.append({choice: tb})
            pass

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
                        type=int, default=20)

    parser.add_argument("--row",
                        help="Number of rows in the base version",
                        type=int, default=1000)

    parser.add_argument("--ver",
                        help="Number of versions to generate",
                        type=int, default=100)

    parser.add_argument("--bfactor",
                        help="Workflow Branching factor, 0.1 is linear, 100 is star-like",
                        type=float, default=5.0)

    parser.add_argument("--matfreq",
                        help="Materialization frequency, i.e. how many operations before writing out an artifact",
                        type=int, default=1)

    parser.add_argument("--npp",
                        help="Generate Merges, Groupbys and Pivots",
                        type=bool, default=False)

    options = parser.parse_args(args)

    return options


def main(args=sys.argv[1:]):

    options = setup_arguments(args)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    out_dir = options.output_dir+'/'+timestr+'/'
    os.makedirs(out_dir, exist_ok=True)
    csv_dir = out_dir+'artifacts/'
    os.makedirs(csv_dir, exist_ok=True)


    dataset, errors = generate_dataset((options.row,options.col),options.ver,out_dir,scale=options.bfactor, gt_prefix=timestr, matfreq=options.matfreq, npp=options.npp)

    print("Errors: ", errors)


if __name__ == '__main__':
    main()
