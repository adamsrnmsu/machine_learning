'''
:Name: Ryan C. Adams
:Course: CS487 Applied Machine Learning I

Usage of the following sklearn classifers on 

:Datasets: 
    i. Digits (Sklearn)
    'http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html'
    ii. Timeseries archive.ics.uci.edu

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
        sys.exit()


    x_dat_train, x_dat_test, y_targ_train, y_targ_test = train_test_split(
        x_dat, y_targ, test_size=0.33, random_state=25)

    report_list =[]

    #perceptron
    record_dict = run_perceptron(
        x_dat_train, x_dat_test, y_targ_train, y_targ_test)
    report_list.append(record_dict)

    #decesion tree
    record_dict = run_dec_tree(
        x_dat_train, x_dat_test, y_targ_train, y_targ_test)
    report_list.append(record_dict)

    #knn
    record_dict = run_knn(
        x_dat_train, x_dat_test, y_targ_train, y_targ_test)
    report_list.append(record_dict)

    #logistic_regression
    record_dict = run_log_reg(
        x_dat_train, x_dat_test, y_targ_train, y_targ_test)
    report_list.append(record_dict)

    # linear svm
    record_dict = run_linear_svm(
        x_dat_train, x_dat_test, y_targ_train, y_targ_test)
    report_list.append(record_dict)

    record_dict = run_rbf_svm(
        x_dat_train, x_dat_test, y_targ_train, y_targ_test)
    report_list.append(record_dict)

    # reporting
    df = pd.DataFrame(report_list)
    df = df.set_index('name')
    display_results(df)


if __name__ == '__main__':
    main()
