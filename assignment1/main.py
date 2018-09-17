'''
:Name: Ryan C. Adams
:Course: CS487_Applied_Machine_Learning
:Implementation: main
driver program for adaline, perceptron, and sgd
'''

import pandas as pd
import numpy as np
import collections
import sys

import preprocess as pre 
from perceptron import Perceptron as perc

def file_handler(file_name):
    '''
    takes a file name from command line sys.argv[2] and
    checks to see if it is:
    correct_file_ending: 'csv'
    not_empty:
    '''
    try:
        file = sys.argv[2]
        ending = file.split(".")
        assert ending[len(ending)-1] == "csv"
    except ValueError as error:
        print('file must end with .csv')
        print('exiting program')
        sys.exit()
    try:
        df = pd.read_csv(file_name)
    except ValueError as error:
        print('The file you entered was not able to be read in')
        print('Please try entering again')
        print('exiting program')
        sys.exit()
    df = pd.read_csv(file_name)
    if df.empty:
        print('The file resulted in an empty dataframe')
        print('Please try entering again')
        print('exiting program')
        sys.exit()

    return df

def algorithm_handler(algorithm, algorithms):
    '''
    checks to see if algorithm is in the implimented list of alogrithms
    param:algorithm: algorithm passed in at command line
    param:alogrithms: list of implemented algorithms
    '''
    if algorithm not in algorithms:
        print('The algorithm you entered is not in the implemented algorithms list')
        print('Please try entering again')
        print('exiting program')
        sys.exit()
    return algorithm

def main():
    algorithms = ['perceptron', 'adaline','sgd']
    algorithm = algorithm_handler(sys.argv[1], algorithms)
    file_name = sys.argv[2]
    raw_data = file_handler(sys.argv[2])
    try:
        eta = float(sys.argv[3])
    except IndexError:
        print("eta value not passed in using default = .01")
        eta = .01
    try:
        itrs = int(sys.argv[4])
    except IndexError:
        print("itr value not passed in using default = 10")
        itrs = 10

    if algorithm == 'perceptron' and file_name == 'iris.csv':
        #preprocess
        row_vectors = pre.create_row_vectors(raw_data, 0, 100, [0,2])
        col_vectors = pre.create_col_vectors(raw_data, 0, 100, [4])
        col_vectors = pre.mod_col_vals(col_vectors,'Iris-setosa', 1, -1)
        #learn predict
        model = perc(eta, itrs)
        model.learn(row_vectors, col_vectors)


    #if algorithm == 'percptron':





if __name__ == '__main__':
    main()
