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

def main():
    x = df.iloc[0:100, [0, 2]].values  # row vector
    # output vectors of size 1, equal to target classification
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)

    # This is just a quick check we have 50 setosa, 50 not setosa
    verify_count = collections.Counter(y)
    print(f"should be 50 ==1 and 50 == -1: {verify_count}")

    #create / train perceptron
    model = Perceptron(eta=.1, iters=10)
    model.learn(x, y)


if __name__ == '__main__':
    main()
