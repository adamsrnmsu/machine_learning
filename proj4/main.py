"""
:Name: Ryan C. Adams
:Course: CS487 Applied Machine Learning I
:Project 4: Regressiongit 

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

import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
import datetime
import argparse
import os
import sys
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
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
    :param housing_data_path: string representing file path to housing dataset
    returns pandas df
    """
    df = pd.read_csv(renewable_data_path)
    return df

def linear_reg_run(data_frame, x_col, y_col):
    x_vector = data_frame.iloc[:, x_col]
    y_vector = data_frame.iloc[:, y_col]

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


def main():
    """
    main function to run regressors
    """
    h_df = housing_to_df('housing.data.txt')
    r_df = renew_to_df('all_breakdown.csv')

    linear_reg_run(h_df,1,2)

if __name__ == '__main__':
    main()
