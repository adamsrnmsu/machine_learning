"""
:Name: Ryan C. Adams
:Course: CS487 Applied Machine Learning I
:Project 4: Regression 

Usage of the following sklearn classifers on 

:Datasets: 
    i. Housing dataset:
       'housing.data.txt' from:
       Canvas course files -> data

    ii. California Renewable Production 2010-2018 dataset:
        'all_breakdown.csv' from:
        https://www.kaggle.com/cheedcheed/california-wind-power-generation-forecasting/data


:Classifiers: 
    i. LinearRegression
    ii. RANSACRegressor
    iii. Ridge
    iv. Lasso
    v. normal equation
    vi. one approach using non-linear regression

"""

import argparse
import pandas as pd
import numpy as np
import pprint
import sys
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import mean_squared_error, r2_score

def housing_to_df(housing_data_path):
    """
    Creates a dataframe from .txt file which is space delimited
    :param housing_data_path: string representing file path to housing dataset
    returns pandas df
    """
    df = pd.read_csv(housing_data_path, header=None, delimiter=r"\s+")
    return df

def renew_to_df(renewable_data_path):
    """
    Creates a dataframe from .txt file which is space delimited
    :param housing_data_path: string representing file path to california
    renewable.
    returns pandas df
    """
    df = pd.read_csv(renewable_data_path)
    return df

def linear_reg_run(data_frame, x_vector, y_vector):
    """
    Linear regression on two columns from a data frame. 
    :param data_frame: pandas data frame
    :x_col: attribute column (guess column)
    :y_col: attribute column (answer column) 
    """

    x_vector = x_vector.values.reshape(-1,1)
    y_vector = y_vector.values.reshape(-1,1)

    reg_obj = LinearRegression().fit(x_vector, y_vector)
    y_pred = reg_obj.predict(x_vector)
    mse = mean_squared_error(y_vector, y_pred)
    r2 = r2_score(y_vector, y_pred)

    print("\n==============================================================")
    print("Linear Regression Results")
    print("==============================================================")

    print("\nCoefficent: ", float(reg_obj.coef_))
    print("Y-intercept: " , float(reg_obj.intercept_))
    print("MSE: ", mse)
    print("R2: ", r2)

def ransac_reg_run(data_frame, x_vector, y_vector):
    """
    ransac regression on two columns from a data frame. 
    :param data_frame: pandas data frame
    :x_vector: attribute column (guess column)
    :y_vector: attribute column (answer column) 
    """

    x_vector = x_vector.values.reshape(-1, 1)
    y_vector = y_vector.values.reshape(-1, 1)

    reg_obj = RANSACRegressor().fit(x_vector, y_vector)
    y_pred = reg_obj.predict(x_vector)
    mse = mean_squared_error(y_vector, y_pred)
    r2 = r2_score(y_vector, y_pred)

    print("\n==============================================================")
    print("RANSCAR Regression Results")
    print("==============================================================")

    print("MSE: ", mse)
    print("R2: ", r2)


def ridge_reg_run(data_frame, x_vector, y_vector, alpha=1.0):
    """
    ridge regression on two columns from a data frame.
    :param data_frame: pandas data frame
    :x_vector: attribute column (guess column)
    :y_vector: attribute column (answer column)
    """

    x_vector = x_vector.values.reshape(-1, 1)
    y_vector = y_vector.values.reshape(-1, 1)

    reg_obj = Ridge(alpha=alpha).fit(x_vector, y_vector)
    y_pred = reg_obj.predict(x_vector)
    mse = mean_squared_error(y_vector, y_pred)
    r2 = r2_score(y_vector, y_pred)

    print("\n==============================================================")
    print("Ridge Regression Results")
    print("==============================================================")

    print("MSE: ", mse)
    print("R2: ", r2)


def lasso_reg_run(data_frame, x_vector, y_vector):
    """
    Lasso regression on two columns from a data frame.
    :param data_frame: pandas data frame
    :x_vector: attribute column (guess column)
    :y_vector: attribute column (answer column)
    """

    x_vector = x_vector.values.reshape(-1, 1)
    y_vector = y_vector.values.reshape(-1, 1)

    reg_obj = Lasso().fit(x_vector, y_vector)
    y_pred = reg_obj.predict(x_vector)
    mse = mean_squared_error(y_vector, y_pred)
    r2 = r2_score(y_vector, y_pred)

    print("\n==============================================================")
    print("Lasso Regression Results")
    print("==============================================================")

    print("MSE: ", mse)
    print("R2: ", r2)


def normal_eq_params(data_frame, x_vector, y_vector):
    """
    normal equation implementation:
    Note I rely heavily on the implmentation examples found at
    https://www.c-sharpcorner.com/article/normal-equation-implementation-from-scratch-in-python/
    :param data_frame: pandas data frame
    :x_vector: attribute column (guess column)
    :y_vector: attribute column (answer column)
    """
    #normal equation 
    #params=(X.T *X)^-1 *X.T * y
    #reshape vectors for continutiy
    x_vector = x_vector.values.reshape(-1, 1)
    y_vector = y_vector.values.reshape(-1, 1)
    
    #implmentation of the normal equation
    #transpose x
    x_trans = x_vector.transpose()

    #X.T *X
    xtrans_dot_x = x_trans.dot(x_vector)
    
    #(X.T *X)^-1
    inverted = np.linalg.inv(xtrans_dot_x)

    #(X.T *X)^-1 *X.T
    invert_dot_trans = inverted.dot(x_trans)

    # Final step
    # (X.T *X)^-1 *X.T * y

    params = invert_dot_trans.dot(y_vector)

    return params

def normal_eq_predict(data_frame, params, x_vector):
    """
    Use pram from normal_eq_params to create a vector of predictions
    Note I rely heavily on the implmentation examples found at
    https://www.c-sharpcorner.com/article/normal-equation-implementation-from-scratch-in-python/
    :param data_frame: pandas data frame
    :params: result from normal_eq_params()
    :x_vector: attribute column (guess column)
    """
    x_vector = x_vector.values.reshape(-1, 1)
    pred = x_vector.dot(params)
    return pred


def normal_eq_score(data_frame, y_vector, pred):
    y_vector = y_vector.values.reshape(-1, 1)
    mse = mean_squared_error(y_vector, pred)
    r2 = r2_score(y_vector, pred)
    return mse, r2

def run_normal_eq(data_frame, x_vector, y_vector):
    params = normal_eq_params(data_frame, x_vector, y_vector)
    pred = normal_eq_predict(data_frame, params, x_vector)
    mse, r2 = normal_eq_score(data_frame, y_vector, pred)

    print("\n==============================================================")
    print("Normal Equation Results")
    print("==============================================================")

    print("MSE: ", mse)
    print("R2: ", r2)


def run_nn_regressor(data_frame, x_vector, y_vector):
    print("\n==============================================================")
    print("Neural MLP Regressor Results")
    print("==============================================================")

    veclist = []
   
    x_vector = x_vector.values.reshape(-1, 1)
    y_vector = y_vector.values.reshape(-1, 1)
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        x_vector, y_vector, test_size=0.5, random_state=0)
    
    veclist = (Xtrain, Xtest, Ytrain, Ytest)

    for vec in veclist:
        vec = np.reshape(vec,(-1,1))

    ml_regressor = MLPRegressor(random_state=0, max_iter = 1000)
    ml_regressor.fit(Xtrain, Ytrain)
    pred = ml_regressor.predict(Xtest)

    mse = mean_squared_error(Ytest, pred)
    r2 = r2_score(Ytest, pred)

    print("MSE: ", mse)
    print("R2: ", r2)

    
def main():
    """
    main function to run regressors
    """
    #h_df = housing_to_df('housing.data.txt')
    #r_df = renew_to_df('all_breakdown.csv')
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--load_from_file",
                        help="specify path to data file", type=str)
    parser.add_argument("-s", "--start",
                        help="column number of where training data (X) starts",
                        type=int)
    parser.add_argument("-t", "--target",
                        help="column number of answer column (Y)", type=int)
    args = parser.parse_args()
    if args.load_from_file:
        f_exists = os.path.isfile(args.load_from_file)
        if f_exists:
            print('exists')
            list_names = args.load_from_file.split(".")
            if any('housing' in x for x in list_names):
                df = housing_to_df(args.load_from_file)
                print("loaded file")
            elif any('all_breakdown' in x for x in list_names):
                df = renew_to_df(args.load_from_file)
                print("loaded file")
            else:
                print("file not expected, is it housing or the all_breakdown.csv?")
                print("exiting")
                sys.exit()
            if df.empty:
                print("Got no data, exiting program")
                sys.exit()
            if not args.start:
                print("You need to specify a X for data")
                sys.exit()
            if not args.target:
                print("You need to specify the Y for regression")
                sys.exit()
            x_dat = df.iloc[:, args.start]
            #x_dat = df.iloc[:, 1:562]
            y_targ = df.iloc[:, args.target]
        else:
            print("the path you entered may be incorrect, could not locate file")
            sys.exit()

    print("does x_column have nans?")
    print(x_dat.isnull().any())
    print("does x_column have nans?")
    print(y_targ.isnull().any())

    # regardless of answer we create a set of indexs of nulls 
    # cat the two lists together
    # take the set so that we have the index of the nans in both sets
    # drop those rows
    y_nulls = y_targ.isnull()
    y_nulls = y_nulls[y_nulls].index
    x_nulls = x_dat.isnull()
    x_nulls = x_nulls[x_nulls].index
    nulls = y_nulls.tolist() + x_nulls.tolist()

    set_nulls = set(nulls)
    y_prefilt = y_targ.drop(set_nulls)
    x_prefilt = x_dat.drop(set_nulls)
    
    linear_reg_run(df, x_prefilt, y_prefilt)
    ransac_reg_run(df, x_prefilt, y_prefilt)
    ridge_reg_run(df, x_prefilt, y_prefilt)
    lasso_reg_run(df, x_prefilt, y_prefilt)
    run_normal_eq(df, x_prefilt, y_prefilt)
    run_nn_regressor(df, x_prefilt, y_prefilt)


if __name__ == '__main__':
    main()
