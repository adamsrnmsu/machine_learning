'''
:Name: Ryan C. Adams
:Course: CS487_Applied_Machine_Learning

Preprocessing steps that were necessary to form the data sets for the models
'''
import pandas as pd
import numpy as np
import collections

def construct_pandas_frame(html, attributes):
    '''
    Creats a pandas dataframe from a csv like data format from a csv
    Also assumes that the header is not in the csv representation nor the index name
    :param: html - string for the location of the csv like data on website
    :param: attributes - list of strings of the given data set in order
    :returns: pandas dataframe
    '''
    df = pd.read_csv(html, header=None)
    df.columns = attributes
    return df


def iris_data():
    html = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    attributes = ['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width', 'class']
    df = construct_pandas_frame(html, attributes)
    df.to_csv('iris.csv', index=False)


def create_row_vectors(df, row_start, row_end, list_cols):
    row_vectors = df.iloc[row_start:row_end, list_cols].values  # row vector
    return row_vectors


def create_col_vectors(df, row_start, row_end, list_cols):
    col_vectors = df.iloc[row_start:row_end, list_cols].values  # row vector
    return col_vectors


def mod_col_vals(col_vectors, targ_val, if_true, if_false):
    col_vectors = np.where(col_vectors == targ_val, if_true, if_false)
    return col_vectors

def accuracy(error, vec_len):
    return (error / vec_len) * 100


def std_rows(row_vecs, col_list):
    '''
    takes row vectors and standardizes
    :param: row_vecs - list of frow vectors created by 
            create_row_vectors
    :param: col_list - list of ints representign columns to be standardized
    :returns: standardized rows
    '''
    std_row_vecs = np.copy(row_vecs)
    for col in col_list:
        std_row_vecs[:, col] = (row_vecs[:, col] - row_vecs[:, col].mean()) / row_vecs[:, col].std()
    return std_row_vecs
