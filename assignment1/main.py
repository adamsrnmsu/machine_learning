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
import matplotlib.pyplot as plt

import preprocess as pre 
from adaline import Adaline
from perceptron import Perceptron
from sgd import SGD

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
        row_vectors = pre.create_row_vectors(raw_data, 0, 150, [0,2])
        col_vectors = pre.create_col_vectors(raw_data, 0, 150, 4)
        col_vectors = pre.mod_col_vals(col_vectors,'Iris-setosa', 1, -1)
        #learn predict
        model = Perceptron(eta, itrs)
        model.learn(row_vectors, col_vectors)
        #report
        x_axis = []
        for error in enumerate(model.errors_list,0):
            accuracy = pre.accuracy(len(row_vectors) - error[1], len(row_vectors))
            x_axis.append(error[0])
            print("iter: " + str(error[0]) + " error: " + str(error[1]) + " accuracy: " + str(accuracy))
        # #plot
        # plt.plot(x_axis, model.errors_list)
        # plt.ylabel("Errors")
        # plt.xlabel("Iterations")
        # plt.xlim([0, itrs+1])
        # plt.ylim([0, max(model.errors_list)+1])
        # plt.show()
    elif algorithm == 'adaline' and file_name == 'iris.csv':
        #preprocess
        row_vectors = pre.create_row_vectors(raw_data, 0, 150, [0, 2])
        std_row_vectors = pre.std_rows(row_vectors,[0,1])
        col_vectors = pre.create_col_vectors(raw_data, 0, 150, 4)
        col_vectors = pre.mod_col_vals(col_vectors, 'Iris-setosa', 1, -1)
        #learn predict
        model2 = Adaline(eta, itrs, )
        model2.learn(std_row_vectors, col_vectors)
        #report
        x_axis = []
        for cost in enumerate(model2.costs, 0):
            accuracy = pre.accuracy(len(row_vectors) - model2.errors_list[cost[0]], len(row_vectors))
            print("iter: " + str(cost[0]) + " cost: " + str(cost[1]) + " accuracy: " + str(accuracy))
            x_axis.append(cost[0])
        #plot
        plt.plot(x_axis, model2.costs)
        plt.ylabel("Costs")
        plt.xlabel("Iterations")
        plt.xlim([0, itrs+1])
        plt.ylim([0, max(model2.costs)+1])
        plt.show()
    elif algorithm == 'sgd' and file_name == 'iris.csv':
        row_vectors = pre.create_row_vectors(raw_data, 0, 150, [0, 2])
        std_row_vectors = pre.std_rows(row_vectors, [0, 1])
        col_vectors = pre.create_col_vectors(raw_data, 0, 150, 4)
        col_vectors = pre.mod_col_vals(col_vectors, 'Iris-setosa', 1, -1)
        #learn predict
        model3 = SGD(eta, itrs)
        model3.learn(std_row_vectors, col_vectors)
        for cost in enumerate(model3.avg_costs, 0):
            print(cost)

    #if algorithm == 'percptron':

if __name__ == '__main__':
    main()
