'''
:Name: Ryan C. Adams
:Course: CS487 Applied Machine Learning I

Usage of the following sklearn classifers on 

:Datasets: 
    i. Digits (Sklearn)
    'http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html'
    ii. Timeseries
        Human Activity Recognition Using Smartphones Data Set:
        'https://www.kaggle.com/mboaglio/simplifiedhuarus/home'

:Classifiers: 
    i. Perceptron
    ii. SVM (linear and non-linear using RBF kernel)
    iii. Decesion tree
    iv. K-nearest neighbor
    v. logistic rgression

'''

import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
import datetime
import time
import argparse
import os
import sys

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

#Todo 1. Support vecotor machines, second data set, argparse


def run_linear_svm(x_dat_train, x_dat_test, y_targ_train, y_targ_test):
    start = datetime.datetime.now()
    lin_svm = svm.SVC(kernel='linear')
    lin_svm.fit(x_dat_train, y_targ_train)
    score = lin_svm.score(x_dat_test, y_targ_test)
    end = datetime.datetime.now()
    total_time = str(end - start)
    times = total_time.split(':')
    time = times[1] + ':' + times[2]
    record_dict = {'name': 'linear_svm'}
    record_dict['time'] = time
    record_dict['accuracy'] = score
    return record_dict


def run_rbf_svm(x_dat_train, x_dat_test, y_targ_train, y_targ_test):
    start = datetime.datetime.now()
    lin_svm = svm.SVC(kernel='rbf')
    lin_svm.fit(x_dat_train, y_targ_train)
    score = lin_svm.score(x_dat_test, y_targ_test)
    end = datetime.datetime.now()
    total_time = str(end - start)
    times = total_time.split(':')
    time = times[1] + ':' + times[2]
    record_dict = {'name': 'rbf_svm'}
    record_dict['time'] = time
    record_dict['accuracy'] = score
    return record_dict


def run_perceptron(x_dat_train, x_dat_test, y_targ_train, y_targ_test):
    start = datetime.datetime.now()
    perc = Perceptron()
    perc.fit(x_dat_train, y_targ_train)
    score = perc.score(x_dat_test, y_targ_test)
    end = datetime.datetime.now()
    total_time = str(end - start)
    times = total_time.split(':')
    time = times[1] + ':' + times[2]
    record_dict = {'name': 'perceptron'}
    record_dict['time'] = time
    record_dict['accuracy'] = score
    return record_dict


def run_dec_tree(x_dat_train, x_dat_test, y_targ_train, y_targ_test):
    start = datetime.datetime.now()
    dec = DecisionTreeClassifier()
    dec.fit(x_dat_train, y_targ_train)
    score = dec.score(x_dat_test, y_targ_test)
    end = datetime.datetime.now()
    total_time = str(end - start)
    times = total_time.split(':')
    time = times[1] +':' + times[2] 
    record_dict = {'name': 'decision tree'}
    record_dict['time'] = time
    record_dict['accuracy'] = score
    return record_dict


def run_knn(x_dat_train, x_dat_test, y_targ_train, y_targ_test):
    start = datetime.datetime.now()
    knn = KNeighborsClassifier()
    knn.fit(x_dat_train, y_targ_train)
    score = knn.score(x_dat_test, y_targ_test)
    end = datetime.datetime.now()
    total_time = str(end - start)
    times = total_time.split(':')
    time = times[1] + ':' + times[2]
    record_dict = {'name': 'knn'}
    record_dict['time'] = time
    record_dict['accuracy'] = score
    return record_dict


def run_log_reg(x_dat_train, x_dat_test, y_targ_train, y_targ_test):
    start = datetime.datetime.now()
    log = LogisticRegression()
    log.fit(x_dat_train, y_targ_train)
    score = log.score(x_dat_test, y_targ_test)
    end = datetime.datetime.now()
    total_time = str(end - start)
    times = total_time.split(':')
    time = times[1] + ':' + times[2]
    record_dict = {'name': 'logistic regression'}
    record_dict['time'] = time
    record_dict['accuracy'] = score
    return record_dict

def display_results(data_frame):
    '''
    Print Helper function
    '''
    print('\n=====================================================')
    print('Generating report for Classifiers')
    print("Note times are in the form min:second.nanoseconds")
    print('=====================================================\n')
    print(data_frame)
    print("\n")

    
def main():
    '''
    main function
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--load_digits", 
                        help="Loads data from digits library",
                        action="store_true")
    parser.add_argument("-s", "--start",
                        help="column number of where training data (x) starts",
                        type=int)
    parser.add_argument("-e", "--end",
                        help="column number of where training data (x) ends",
                        type=int)
    parser.add_argument("-t", "--target",
                        help="column number of the target",
                        type=int)
    parser.add_argument("-f", "--load_from_file", 
                        help="specify path to data file", type=str)
    args = parser.parse_args()
    if args.load_digits:
        digits = load_digits()
        #data that to be learned
        x_dat = digits.data
        #digit targets vector of number
        y_targ = digits.target

    if args.load_from_file:
        f_exists = os.path.isfile(args.load_from_file)
        print(args.load_from_file)
        print(type(args.load_from_file))
        print(f_exists)
        if f_exists:
            print('exists')
            df = pd.read_csv(args.load_from_file)
            if df.empty:
                print("Got no data, exiting program")
                sys.exit()
            if not args.start:
                print("You need to specify a start for your attribute training data")
                sys.exit()
            if not args.end:
                print("You need to specify the end for your attribute training data")
                # print("Remember that the slice is not inclusive of the first number")
                # print("So if column two is the actual start")
                sys.exit()
            if not args.target:
                print("You need to specify the target for classification")
                sys.exit()
            x_dat = df.iloc[:, args.start: args.end]
            #x_dat = df.iloc[:, 1:562]
            y_targ = df.iloc[:, args.target]
            print(y_targ)
            print(x_dat)
        else:
            print("the path you entered may be incorrect, could not locate file")
            sys.exit()

    x_dat_train, x_dat_test, y_targ_train, y_targ_test = train_test_split(
        x_dat, y_targ, test_size=0.33, random_state=25)

    func_list = [run_perceptron, run_dec_tree, run_knn, run_log_reg, 
                 run_linear_svm, run_rbf_svm]

    report_list = []
    
    for func in func_list:
        record_dict = func(x_dat_train, x_dat_test, y_targ_train, y_targ_test)
        report_list.append(record_dict)

    # reporting
    df = pd.DataFrame(report_list)
    df = df.set_index('name')
    display_results(df)


if __name__ == '__main__':
    main()
