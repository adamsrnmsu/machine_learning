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

def pre_process_iris(iris_data, std=False):
    '''
    preprocsses iris data set
    :param: std - if true standardizes the row_vectors
    returns row_vectors, col_vectors
    '''
    row_vectors = pre.create_row_vectors(iris_data, 0, 150, [0, 2])
    col_vectors = pre.create_col_vectors(iris_data, 0, 150, 4)
    col_vectors = pre.mod_col_vals(col_vectors, 'Iris-setosa', 1, -1)
    # if standardize is ture returns standardized row_vectors
    if std == True:
        row_vectors = pre.std_rows(row_vectors, [0, 1])
    return row_vectors, col_vectors

def perceptron_model(row_vectors, col_vectors, eta, iterations):
    '''
    creates a perceptron model
    '''
    model = Perceptron(eta, iterations)
    model.learn(row_vectors, col_vectors)
    return model

def adaline_model(row_vectors, col_vectors, eta, iterations):
    '''
    creates a adaline model
    '''
    model = Adaline(eta, iterations)
    model.learn(row_vectors, col_vectors)
    return model


def sgd_model(row_vectors, col_vectors, eta, iterations):
    '''
    creates a sgd model
    '''
    model = SGD(eta, iterations)
    model.learn(row_vectors, col_vectors)
    return model


def print_perceptron(model, row_vectors):
    '''
    printer helper function for perceptron
    '''
    for error in enumerate(model.errors_list, 0):
        accuracy = pre.accuracy(
            len(row_vectors) - error[1], len(row_vectors))
        print("iter: " + str(error[0]) + " error: " +
                str(error[1]) + " accuracy: " + str(accuracy))


def print_adaline(model, row_vectors):
    '''
    printer helper function for adaline
    '''
    for cost in enumerate(model.costs, 0):
        accuracy = pre.accuracy(
            len(row_vectors) - model.errors_list[cost[0]], len(row_vectors))
        print("iter: " + str(cost[0]) + " cost: " +
                str(cost[1]) + " accuracy: " + str(accuracy))


def print_sgd(model, row_vectors):
    for cost in enumerate(model.avg_costs, 0):
        accuracy = pre.accuracy(
            len(row_vectors) - model.errors_list[cost[0]], len(row_vectors))
        print("iter: " + str(cost[0]) + " cost: " +
                str(cost[1]) + " accuracy: " + str(accuracy))


def perceptron_plot_helper(perceptron):
    '''
    plot function for perceptron
    '''
    x_axis = list(range(0, len(perceptron.errors_list)))
    plt.plot(x_axis, perceptron.errors_list)
    plt.ylabel("Errors")
    plt.xlabel("Iterations")
    plt.show()


def adaline_plot_helper(adaline):
    '''
    plot function for perceptron
    '''
    x_axis = list(range(0, len(adaline.errors_list)))
    plt.plot(x_axis, adaline.costs)
    plt.ylabel("Costs")
    plt.xlabel("Iterations")
    plt.show()

def sgd_plot_helper(sgd):
    x_axis = list(range(0, len(sgd.avg_costs)))
    plt.plot(x_axis, sgd.avg_costs)
    plt.ylabel("Avg_Costs")
    plt.xlabel("Iterations")
    plt.show()


def main():
    #take in command line arguments
    #python3 main.py perceptron iris.csv .01 100 False False
    algorithms = ['perceptron', 'adaline','sgd']
    algorithm = algorithm_handler(sys.argv[1], algorithms)
    file_name = sys.argv[2]
    raw_data = file_handler(sys.argv[2])
    eta = float(sys.argv[3])
    iters = int(sys.argv[4])
    std_data_flag = bool(sys.argv[5])
    graph_data_flag = sys.argv[6]

    #preprocess of data

    if file_name == 'iris.csv':
        row_vectors, col_vectors = pre_process_iris(raw_data, std_data_flag)

    if algorithm == 'perceptron':
        #learn predict
        perceptron = perceptron_model(row_vectors, col_vectors, eta, iters)
        #print results
        print_perceptron(perceptron, row_vectors)
        #plot
        if graph_data_flag == 'True':
            perceptron_plot_helper(perceptron)
    elif algorithm == 'adaline':
        #learn predict
        adaline = adaline_model(row_vectors, col_vectors, eta, iters)
        #print results
        print_adaline(adaline, row_vectors)
        #plot
        if graph_data_flag == 'True':
            adaline_plot_helper(adaline)
    elif algorithm == 'sgd':
        #learn predict
        sgd = sgd_model(row_vectors, col_vectors, eta, iters)
        #print result
        print_sgd(sgd, row_vectors)
        #plot
        if graph_data_flag == 'True':
            sgd_plot_helper(sgd)

if __name__ == '__main__':
    main()
