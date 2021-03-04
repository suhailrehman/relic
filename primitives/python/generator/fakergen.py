import pandas as pd
import numpy as np
import sys
import traceback
from faker import Faker
import networkx as nx
import argparse
import time
import os
import pickle
import json

from exceptions import *
from lineage import LineageTracker

from tqdm.auto import tqdm

from pyvis.network import Network
from networkx.drawing.nx_agraph import graphviz_layout


def compute_jaccard_DF(df1,df2, pk_col_name=None):

    try:
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

    except ValueError as e:
        return 0.0


class FakerVersionGenerator:

    def __init__(self, shape=(8,20), out_directory='dataset/',
                 num_base_versions=1, scale=10., gt_prefix='dataset', npp=False, matfreq=1):

        self.functions = self.load_function_dict()
        self.inv_functions = self.inv_function_dict()


        # Keep track of all generated columns so that we dont repeat.
        self.groupby_cols = set()
        self.pivot_indices = set()
        self.pivot_cols = set()
        self.pivot_vals = set()
        self.generated_cols = set()


        self.fake = Faker()

        rowsize, colsize = shape

        self.dataset = []
        self.dataset_metadata = []

        # Probablistic columns selection:
        with open('sources/config_dict.json', 'r') as fp:
            self.config_dict = json.load(fp)
        self.counts = pickle.load(open("sources/counts.pkl", "rb"))


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

        self.lastargs = {}
        self.op_equv_map = {}

        self.npp_chain_threshold = 1
        self.MIN_DATA_THRESHOLD = 0.5






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

    def get_next_label(self):
        return str(len(self.dataset))

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def search_lower(self, row_floor, card_choice, col_type=None):
        c = card_choice
        while c >= 0:
            if str(c) not in self.config_dict['prob_dict'][str(row_floor)]:
                pass
            elif col_type is not None:
                if col_type in self.config_dict['prob_dict'][str(row_floor)][str(c)]:
                    cols = [x for x in self.config_dict['prob_dict'][str(row_floor)][str(c)][col_type] if
                            x not in self.generated_cols]
                    if cols:
                        return cols
            else:
                cols = None
                try:
                    cols = [item for sublist in self.config_dict['prob_dict'][str(row_floor)][str(card_choice)].values() for
                            item in sublist if item not in self.generated_cols]
                except KeyError as e:
                    pass
                if cols:
                    return cols
            c -= 1

        # Did not find the requested type at all
        return None

    def search_upper(self, row_floor, card_choice, col_type=None, num_bins=10):
        c = card_choice
        #print(col_type, 'entry', c)
        while c < num_bins:
            if str(c) not in self.config_dict['prob_dict'][str(row_floor)]:
                pass
            elif col_type is not None:
                if col_type in self.config_dict['prob_dict'][str(row_floor)][str(c)]:
                    cols = [x for x in self.config_dict['prob_dict'][str(row_floor)][str(c)][col_type] if x not in self.generated_cols]
                    if cols:
                        print(col_type, 'exit', c)
                        return cols
            else:
                #print(col_type, 'exit', c)
                cols = None
                try:
                    cols = [item for sublist in self.config_dict['prob_dict'][str(row_floor)][str(card_choice)].values()
                            for
                            item in sublist if item not in self.generated_cols]
                except KeyError as e:
                    pass
                if cols:
                    return cols
            c += 1

        # Did not find the requested type at all
        #print(col_type, 'final exit', c)
        return None

    def select_column_from_dist(self, num_rows, col_type=None):
        # First try to get column of requested type
        ybins = [11, 101, 1001, 10001, 100001, 1000001]

        row_index = min(min(np.digitize(num_rows, ybins) - 1, len(ybins) - 1), self.config_dict['row_sizes'][-1])
        row_floor = self.config_dict['row_sizes'][row_index]

        card_prob = self.counts.T[row_index] / self.counts.T[row_index].sum()
        card_choice = np.random.choice(range(len(card_prob)), 1, p=card_prob)[0]

        #print('Row Count:', row_floor, 'card_choice:', card_choice)

        col_options = self.search_lower(row_floor, card_choice, col_type=col_type)
        if not col_options:
            col_options = self.search_upper(row_floor, card_choice, col_type=col_type, num_bins=len(card_prob))

        if not col_options:
            # Error
            #print("COULD NOT FIND OPTION IN SELECTED ROW BRACKET")
            return None

        return np.random.choice(col_options, 1)[0]

    def select_new_cols_flat(self, num_cols, num_rows=1000):
        all_cols = []
        for _ in range(num_cols):
            s_col = None
            i, max_tries = 0, 10
            while s_col is None and i < max_tries:
                s_col = self.select_column_from_dist(num_rows, col_type=None)
                self.generated_cols.add(s_col)
                i += 1
            if s_col is not None:
                all_cols.append(s_col)
        return all_cols


    def select_new_cols(self, group, num_cols, repeat=False, with_prob=True, num_rows=1000):
        if with_prob:
            all_cols = []
            for _ in range(num_cols):
                s_col = None
                i, max_tries = 0, 10
                while s_col is None and i < max_tries:
                    s_col = self.select_column_from_dist(num_rows, col_type=group)
                    self.generated_cols.add(s_col)
                    i += 1
                if s_col is not None:
                    all_cols.append(s_col)
            return all_cols
        else:
            selection_list = [c for c in self.functions[group] if c not in self.generated_cols]
            try:
                return np.random.choice(selection_list, num_cols, replace=repeat).tolist()
            except ValueError as e:
                raise pd.errors.EmptyDataError

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

    def select_rand_dataset(self, for_merge=False, min_df_size=5):
        tries = 0

        while tries < len(self.dataset):
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

            if len(self.dataset[choice].index) > min_df_size:
                return choice
            tries += 1

        raise pd.errors.EmptyDataError

    def get_column_counts(self, num_cols, num_cat=4):
        four_rands = np.random.multinomial(num_cols, np.ones(num_cat)/num_cat, size=1)[0]
        difference = 0
        function_keys = [x for x in self.functions.keys()]
        function_keys.remove('joinable')

        for i, category in enumerate(function_keys):
            if four_rands[i] > len(self.functions[category]):
                difference += four_rands[i] - len(self.functions[category])
                four_rands[i] = len(self.functions[category])

        for i, category in enumerate(function_keys):
            to_add = min(difference, len(self.functions[category])-four_rands[i])
            four_rands[i] += to_add
            difference -= to_add

        return four_rands



    def generate_base_df(self, num_cols=8, num_rows=20, exclusions=None, atleast_one_pk=False,
                     repeat_cols=False, seed=None, index_col=None,
                     join_cols=None):

        #TODO: Seed Selection
        #TODO: Index Column (Rely on Autonumbered Index for now)

        #TODO: Customizable Column Groups
        four_rands = self.get_column_counts(num_cols, num_cat=3)
        #print('Selection Config: 1 +', four_rands)
        num_joinable = 1
        num_groupable = four_rands[0]
        num_numeric = four_rands[1]
        num_string = four_rands[2]

        selected_cols = []

        '''
        selected_cols.extend(self.select_new_cols('joinable', num_joinable))
        selected_cols.extend(self.select_new_cols('groupable', num_groupable))
        selected_cols.extend(self.select_new_cols('numeric', num_numeric))
        selected_cols.extend(self.select_new_cols('string', num_string))
        '''

        # Cardinality Enforcement
        '''
        selected_cols.extend(self.select_new_cols('joinable', num_joinable, with_prob=True, num_rows=num_rows))
        selected_cols.extend(self.select_new_cols('groupable', num_groupable, with_prob=True, num_rows=num_rows))
        selected_cols.extend(self.select_new_cols('numeric', num_numeric, with_prob=True, num_rows=num_rows))
        selected_cols.extend(self.select_new_cols('string', num_string, with_prob=True, num_rows=num_rows))
        '''

        # Cardinality Enforcement (Flat column type hierarchy)
        selected_cols = None
        i, max_retries = 0, 10

        while selected_cols is None and i < max_retries:
            selected_cols = self.select_new_cols_flat(num_cols, num_rows=num_rows)
            if exclusions:
                for e in exclusions:
                    if e in selected_cols:
                        selected_cols.remove(e)
            i += 1

        print("Base DF: ", selected_cols)
        series = []

        for col in selected_cols:
            col_type = self.inv_functions[col]
            if col_type == 'numeric':
                gen_col = pd.Series((self.fake.format(col) for _ in range(num_rows)), dtype='float64')
            else:
                gen_col = pd.Series((self.fake.format(col) for _ in range(num_rows)))
            series.append(gen_col)

            self.generated_cols.add(col)

        return pd.concat(series, axis=1, keys=selected_cols)

    def apply_op(self, op_function, **kwargs):
        similar_versions = []

        if op_function == self.merge:  # Merge is special case

            #Select random dataset as left side of merge
            choice = self.select_rand_dataset()
            df1 = self.dataset[choice]

            #Select a join column
            #TODO: Handle tables without any join column to being with
            join_column = self.select_rand_col_group(df1, 'joinable', 1)[0]
            print('Join Column: ', join_column)
            duplicate_columns = set(df1.columns)
            join_values = set(df1[join_column].values)
            try:
                df2 = self.generate_base_df(num_rows=len(join_values), exclusions=duplicate_columns)
            except ValueError as e:
                raise pd.errors.EmptyDataError

            #Add the join column to new df if not already present
            # TODO: Generate any length df2 and pad with newly generated faker values
            #if join_column not in df2.columns.values:
            df2[join_column] = list(join_values)

            # Append newly generated merge table as dataset item:
            merge_tbl_ver = str(len(self.dataset))
            self.lineage.new_item(merge_tbl_ver, df2)
            self.dataset.append(df2)

            # Perform the merge and save result as new version

            new_df = self.merge(df1, df2, on=join_column)#.dropna()

            if type(new_df) is not pd.DataFrame or new_df.empty:
                raise pd.errors.EmptyDataError


            for i, d in enumerate(tqdm(self.dataset)):
                try:
                    if i == choice:
                        continue
                    if compute_jaccard_DF(d, new_df) == 1.0:
                        self.lastargs = {}
                        raise TooSimilarException
                    other_new_df = self.merge(d, df2, on=join_column)#.dropna()
                except Exception as e:
                    continue

                # TODO: Check the validity of this when matfreq neq 1
                if type(other_new_df) is not pd.DataFrame or other_new_df.empty:
                    continue
                elif compute_jaccard_DF(other_new_df, new_df) == 1.0:
                    similar_versions.append(i)

            self.lineage.new_item(str(len(self.dataset)), new_df)
            self.dataset.append(new_df)
            self.lineage.link(str(choice), self.get_last_label(),
                              str(op_function.__name__), args={'key': join_column})
            self.lineage.link(merge_tbl_ver, self.get_last_label(),
                              str(op_function.__name__), args={'key': join_column})

            if similar_versions:
                self.op_equv_map[(self.lastmatchoice, self.get_last_label())] = similar_versions

        else:
            if self.opcount == 0:
                choice = self.select_rand_dataset()
                self.lastmatchoice = choice
                base_df = self.dataset[choice]

                # Check NPP across the chain:
                if op_function in [self.pivot, self.groupby]:
                    print('Chain check for pivot and groupby')
                    num_previous = self.find_op_num_in_chain(choice, str(op_function.__name__))
                    if num_previous >= self.npp_chain_threshold:
                        print('Chain threshold exceeded', num_previous)
                        raise TooSimilarException

            else:
                base_df = self.currentdf

            self.lastargs = {}
            new_df = op_function(base_df, **kwargs)

            if type(new_df) is not pd.DataFrame or new_df.empty:
                self.lastargs = {}
                raise pd.errors.EmptyDataError

            #NaN Check < 50%
            data_ratio = new_df.count().sum() / np.product(new_df.shape)
            if data_ratio < self.MIN_DATA_THRESHOLD:
                print('Too many NANS')
                self.lastargs = {}
                raise pd.errors.EmptyDataError

            # Check if newly generated DF is identical (cell-wise to previous):
            if compute_jaccard_DF(new_df, base_df) == 1.0:
                self.lastargs = {}
                raise TooSimilarException

            #new_df = new_df.dropna()
            self.currentdf = new_df
            self.opcount +=1

            if self.opcount == self.matfreq:
                # Check for any dropped columns between the old and new dataframe if not pivot
                '''
                if op_function not in [self.pivot, self.dropcol]:
                    missing_cols = set(base_df) - set(new_df)
                    if missing_cols:
                        print('Missing cols as a result of this new operation', str(op_function.__name__), missing_cols)
                        middle_df = base_df.drop(list(missing_cols), axis=1)
                        print(middle_df)
                        self.lineage.new_item(str(len(self.dataset)), middle_df)
                        self.dataset.append(middle_df)
                        self.lineage.link(str(self.lastmatchoice), self.get_last_label(), 'dropcol')
                        self.lastmatchoice = self.get_last_label()
                '''

                for i, d in enumerate(tqdm(self.dataset)):
                    if i == self.lastmatchoice:
                        continue
                    if compute_jaccard_DF(d, new_df) == 1.0:
                        self.lastargs = {}
                        raise TooSimilarException
                    try:
                        other_new_df = op_function(d, **kwargs)
                    except Exception as e:
                        continue

                    # TODO: Check the validity of this when matfreq neq 1
                    if type(other_new_df) is not pd.DataFrame or other_new_df.empty:
                        continue
                    elif compute_jaccard_DF(other_new_df, new_df) == 1.0:
                        similar_versions.append(i)


                self.lineage.new_item(str(len(self.dataset)), new_df)
                self.dataset.append(new_df)
                self.lineage.link(str(self.lastmatchoice), self.get_last_label(),
                                  str(op_function.__name__), args=self.lastargs)
                self.opcount = 0
                self.currentdf = None
                if similar_versions:
                    self.op_equv_map[(self.lastmatchoice, self.get_last_label())] = similar_versions

    def select_rand_aggregate(self):
        return np.random.choice(['min', 'max', 'sum', 'mean', 'count'], 1)[0]

    def assign_numeric(self, df):
        if 'col' in self.lastargs:
            col = self.lastargs['col']
            random_scalar = self.lastargs['random_scalar']
            new_col_name = self.lastargs['new_col_name']

            if new_col_name in df.columns or col not in df.columns:
                print("Assigned Column already exists / Original column does not exist")
                return None

        else:
            col = self.select_rand_col_group(df, 'numeric', 1)[0]
            random_scalar = np.random.randint(1, 100, 2)
            new_col_name = col+'__'+str(random_scalar[0])+'x+'+str(random_scalar[1])

            if new_col_name in df.columns: # or new_col_name in self.generated_cols:
                print("Assigned Column already exists")
                return None

            self.lastargs.update({
                'col': col,
                'random_scalar': random_scalar,
                'new_col_name': new_col_name
            })

        print("Applying linear combination", random_scalar, "to column", col)

        self.generated_cols.add(new_col_name)

        #if random_func == 'pow':
        #    return df.assign(**{new_col_name: lambda x: np.power(x[col], random_scalar)})
        #elif random_func == 'exp':
        #    return df.assign(**{new_col_name: lambda x: np.exp(x[col])})
        #elif random_func == 'log':
        #    return df.assign(**{new_col_name: lambda x: np.log(x[col])})
        #else:
        #    return df.assign(**{new_col_name: lambda x: np.nancumsum(x[col].astype('float64'))})

        # replace with ax+b transformation

        return df.assign(**{new_col_name: lambda x: x[col]*random_scalar[0]+random_scalar[1]})

    def assign_string(self, df):
        if 'new_col_name' in self.lastargs:
            col = self.lastargs['col']
            new_col_name = self.lastargs['new_col_name']

            if new_col_name in df.columns or col not in df.columns:
                print("Assigned Column already exists/ Old Column not in DF")
                return None

        else:

            col = self.select_rand_col_group(df, 'string', 1)[0]
            new_col_name = col+'__swapcase'

            if new_col_name in df.columns or new_col_name in self.generated_cols:
                print("Assigned Column already exists")
                return None

            self.lastargs.update({
                'col': col,
                'new_col_name': new_col_name
            })

            self.generated_cols.add(new_col_name)

        try:
            if type(df[col].iloc[0]) == list:
                return df.assign(**{new_col_name: lambda x: x[col].apply(lambda y: ",".join(y).swapcase())})
            else:
                return df.assign(**{new_col_name: lambda x: x[col].astype(str).apply(lambda y: y.swapcase())})
        except IndexError as e:
            print(df, col)
            self.lastargs.pop('col')
            self.lastargs.pop('new_col_name')
            return None

    def assign(self, df):
        if 'string_or_numeric' in self.lastargs:
            string_or_numeric = self.lastargs['string_or_numeric']
        else:
            #string_or_numeric = np.random.choice(['string', 'numeric'], 1)[0]
            self.lastargs['string_or_numeric'] = 'numeric'

        #if string_or_numeric == 'string':
        #    return self.assign_string(df)
        #else:
        # Numeric assign function only
        return self.assign_numeric(df)

    def groupby(self, df):
        if 'col' in self.lastargs:
            col = self.lastargs['col']
            func = self.lastargs['func']
            if any(c not in df.columns for c in col):
                return None
        else:
            # TODO: Ensure groupable columns exist in dataframe
            group_size = np.random.randint(1,high=4)
            col = self.select_rand_col_group(df, 'groupable', group_size)
            #if col[0] not in self.groupby_cols:
            #    self.groupby_cols.add(col[0])
            #else:
            #    return None # Cancel this groupby
            func = self.select_rand_aggregate()
            self.lastargs['col'] = col
            self.lastargs['func'] = func

        try:
            print("Grouping By: ", col, 'aggregation: ', func)
            method = getattr(df.groupby(col), func)
            new_df = method().reset_index()  # TODO: Verify drop behavior
        except pd.core.base.DataError as e:
            print('Cannot apply selected groupby:', e)
            return None

        if len(new_df.index) == len(df.index):
            return None  # No Contraction: GroupBy doesnt make sense here.

        if not set(new_df) - set(col):  # Atleast one aggreate column generated
            return None

        return new_df

    def iloc(self, df):
        # Select random row slice
        # Maybe select at-least 10% of the rows?
        if len(df.index) < 10:
            return None

        if 'num1' in self.lastargs:
            num1 = self.lastargs['num1']
            num2 = self.lastargs['num2']
            if num2 >= len(df.index):
                return None
        else:
            stop = False
            while not stop:
                num1 = np.random.randint(0, len(df.index))
                num2 = np.random.randint(num1, len(df.index))
                print("Random iloc range %:", (num2 - num1) / len(df.index))
                if (num2 - num1) / len(df.index) >= 0.1:
                    stop = True

            self.lastargs['num1'] = num1
            self.lastargs['num2'] = num2

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
        if 'frac' in self.lastargs:
            frac = self.lastargs['frac']
        else:
            frac = self.get_rand_percentage()

        self.lastargs['frac'] = frac

        return df.sample(frac=frac)

    def sort_values(self, df):
        col = self.select_rand_col_group(df, 'numeric', 1)[0]
        choice = np.random.choice([True, False], 1)[0]
        if col:
            return df.sort_values(by=col, ascending=choice)
        else:
            return None

    def point_edit(self, df):
        if 'old_value' in self.lastargs:
            col = self.lastargs['col']
            old_value = self.lastargs['old_value']
            new_value = self.lastargs['new_value']
            if col not in df.columns or old_value not in df[col].values:
                return None
        else:
            col = self.select_rand_cols(df, 1)[0]
            print("Selected Column", col)
            if not col:
                return None

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
            while new_value == old_value:  # In case we end up with same value
                old_value = np.random.choice(list(colvalues), 1)[0]
                new_value = self.fake.format(colname)

            self.lastargs = {
                'col': col,
                'old_value': old_value,
                'new_value': new_value
            }

        new_df = df.copy()
        print("Replacing", col, "value", old_value, "with", new_value)
        new_df.loc[new_df[col] == old_value, col] = new_value
        return new_df

    def dropcol(self, df):
        if 'col' in self.lastargs:
            col = self.lastargs['col']
        else:
            col = self.select_rand_cols(df, 1)[0]
            print("Dropping column", col)

        if col and col in df.columns:
            self.lastargs['col'] = col
            new_df = df.copy()
            new_df = df.drop(col, axis=1)
            return new_df

        else:
            return None

    def merge(self, df1, df2, on=None):
        return df1.merge(df2, on=on, suffixes=['__x', '__y'])

    def pivot(self, df):
        if 'index' in self.lastargs:
            index = self.lastargs['index']
            column = self.lastargs['column']
            numeric = self.lastargs['numeric']
            if not all(col in df.columns for col in [index, column, numeric]):
                return None

        else:
            index, column = self.select_rand_col_group(df, 'groupable', 2)
            # if index in self.pivot_indices or column in self.pivot_cols:
            #    return None
            numeric = self.select_rand_col_group(df, 'numeric', 1)[0]
            # if numeric in self.pivot_vals:
            #    return None
            print('Pivoting using index:', index, ' column:', column, 'and values:', numeric)

        newdf = df.pivot_table(index=index, columns=column, values=numeric, aggfunc=max)
        newdf.columns = newdf.columns.map(str)
        newdf.rename(columns={x: str(x)+'__pivoted' for x in newdf.columns})

        self.lastargs['index'] = index
        self.lastargs['column'] = column
        self.lastargs['numeric'] = numeric
        self.pivot_indices.add(index)
        self.pivot_cols.add(column)
        self.pivot_vals.add(numeric)

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
                    #self.merge,
                    self.groupby,
                    self.pivot
                ]
            )

        '''
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
        '''
        return np.random.choice(operations, 1)[0]

    def write_graph_files(self):
        def csv_mapping(x):
            return x+'.csv'

        csv_graph = nx.relabel_nodes(self.lineage.graph, csv_mapping)

        nx.write_gpickle(csv_graph, self.out_directory+self.gt_prefix+'_gt_fixed.pkl')
        nx.write_edgelist(csv_graph, self.out_directory+self.gt_prefix+'_gt_edgelist.txt')

        with open(self.out_directory+self.gt_prefix+'_gt_similar_nodes.pkl', 'wb') as handle:
            pickle.dump(self.op_equv_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def find_op_num_in_chain(self, source, op_name):
        ops_in_chain = [x[2]['operation'] for x in self.lineage.graph.edges(str(source), data=True)]
        rev_graph = self.lineage.graph.reverse()
        edges = [x for x in nx.dfs_edges(rev_graph, source=str(source))]
        ops_in_chain.extend([rev_graph[u][v]['operation'] for u, v in edges])
        return len([x for x in filter(lambda x: x == op_name, ops_in_chain)])

    def draw_interactive_graph(self):
        g = self.lineage.graph
        df_dict = {str(i): df for i, df in enumerate(self.dataset)}
        # , bgcolor="#222222", font_color="white",
        nb_net = Network(height="750px", width="100%", notebook=True)

        # g = get_graph(RESULT_DIR, selected_nb).to_undirected()
        # g_inferred = get_graph_edge_list(RESULT_DIR, selected_nb, metric)
        # df_dict = similarity.load_dataset_dir(RESULT_DIR + selected_nb + '/artifacts/', '*.csv', index_col=0)

        # print(df_dict)

        if '0.csv' not in df_dict:
            try:
                root_node = [x for x in nx.topological_sort(g)][0]  # TODO: Check more than one root issues
            except nx.exception.NetworkXUnfeasible as e:
                print("ERROR: Cycle in Graph")
                root_node = list(df_dict.keys())[0]
                pass
        else:
            root_node = '0.csv'

        pos = graphviz_layout(g, root=root_node, prog='dot')

        # nb_data = pd.DataFrame(similarity.get_all_node_pair_scores(df_dict, g))

        # sources = nb_data['source']
        # targets = nb_data['dest']
        # weights = nb_data['cell']

        # edge_data = zip(sources, targets, {'weight': w for w in weights})

        for edge in g.edges(data=True):
            src = edge[0]
            dst = edge[1]
            w = 1

            src_node_hover_html = df_dict[src].head().to_html() + "<br> Rows:" + str(
                len(df_dict[src])) + " Columns:" + str(
                len(set(df_dict[src])))
            dst_node_hover_html = df_dict[dst].head().to_html() + "<br> Rows:" + str(
                len(df_dict[dst])) + " Columns:" + str(
                len(set(df_dict[dst])))

            # Add Edges
            nb_net.add_node(src, src, x=pos[src][0], y=pos[src][1], physics=False, title=src_node_hover_html)
            nb_net.add_node(dst, dst, x=pos[dst][0], y=pos[dst][1], physics=False, title=dst_node_hover_html)

            nb_net.add_edge(src, dst, value=w, label=g[src][dst]['operation'], physics=False)

        return nb_net




def generate_dataset(shape, n, output_dir, scale=10., gt_prefix='dataset', npp=False, matfreq=1):

    dataset = FakerVersionGenerator(shape, scale=scale, out_directory=output_dir, gt_prefix=gt_prefix, npp=npp, matfreq=matfreq)

    errors = []

    i = 0
    chain_retries = 10

    while i < n-1 and chain_retries > 0:
        retries = 10
        while retries > 0:
            print(retries)
            choice = dataset.select_rand_op()
            try:
                if i == n-2 and choice.__name__ == 'merge':
                    print('Last artifact, cannot do merge here')
                    raise pd.errors.EmptyDataError
                print("Version: "+dataset.get_next_label()+" applying: "+ str(choice.__name__))
                dataset.apply_op(choice)
                if choice.__name__ == 'merge':
                    i += 1  # Extra artifact generated for merges
                i += 1
                chain_retries = 10
                retries = 10
                break
            except pd.errors.EmptyDataError as e:
                print("Empty DF result")
                retries -= 1
                pass
            except ColumnTypeException as e:
                print(e)
                print("Cannot apply operation because of missing column type, skipping")
                retries -= 1
                pass
            except TooSimilarException as e:
                print(e)
                print("Cannot apply operation because generated dataframe is too similar to ones already generated, skipping")
                retries -= 1
                pass
            except Exception as e:
                dataset.lastargs = {}
                print(dataset.lastmatchoice)
                tb = traceback.format_exc()
                errors.append({choice: tb})
                print(dataset.lastargs)
                raise

        if retries == 0:
            dataset.opcount = 0
            dataset.currentdf = None
            chain_retries -= 1

    if chain_retries > 0:
        dataset.write_graph_files()

    print(dataset.op_equv_map)
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

    dataset, errors = generate_dataset((options.row,options.col), options.ver, out_dir,
                                       scale=options.bfactor, gt_prefix=timestr,
                                       matfreq=options.matfreq, npp=options.npp)

    print("Errors: ", errors)


if __name__ == '__main__':
    main()
