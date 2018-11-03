""""
: Name: Ryan C. Adams
: Course: CS487 Applied Machine Learning I
: Project 5: Clustering

Usage of the following sklearn classifers on

: Datasets:
    i. Iris dataset:
       'iris.csv'

    ii. Faulty steel plates dataset:
        'faults.csv' from:
        https://www.kaggle.com/uciml/faulty-steel-plates


: Classifiers:
    i. Kmeans
    ii. Heirarchical from Scipy
    iii. Heirarchical from Scikit
    iv. DBSCAN from scikit
    v. normal equation
    vi. one approach using non-linear regression
"""


import pandas as pd
import numpy as np
import pprint

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def runplot_kmeans(data_frame, col_1, col_2, run_list):
    #combine two columns two create an array of X
    col_1 = 1
    col_2 = 2

    X = data_frame.iloc[:, [col_1, col_2]].values
    SSE_list = []
    iter_list = run_list
    plot_dict = {}
    for num_clusters in iter_list:
        k_clusters = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        sse_val = k_clusters.inertia_
        plot_dict[num_clusters] = sse_val

    print("Results of kmeans cluster")
    print("n_clusters, SSE")
    pprint.pprint(plot_dict)

    plt.figure()
    plt.plot(list(plot_dict.keys()), list(plot_dict.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("Sum Squared Error")
    plt.show()

def main():
    #load datasets
    df_iris = pd.read_csv('iris.csv')
    df_plates = pd.read_csv('faults.csv')

    #lists for plotting
    one_to_10 = list(range(1,11))
    custom = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

    #runplot_kmeans(df_iris, col_1=1, col_2=2, one_to_10)
    #runplot_kmeans(df_plates, col_1=5, col_2=6, run_list=custom)
    print(df_plates)
    print(df_plates.columns)



if __name__ == '__main__':
    main()
