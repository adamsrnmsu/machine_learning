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
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def runplot_kmeans(data_frame, col_1, col_2, run_list, plot=False):
    X = data_frame.iloc[:, [col_1, col_2]].values
    SSE_list = []
    iter_list = run_list
    plot_dict = {}
    for num_clusters in iter_list:
        k_clusters = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
        sse_val = k_clusters.inertia_
        plot_dict[num_clusters] = sse_val
    print("==========================================")
    print("Results of kmeans cluster")
    print("n_clusters, SSE")
    pprint.pprint(plot_dict)
    print("==========================================")

    if plot == True:
        plt.figure()
        plt.plot(list(plot_dict.keys()), list(plot_dict.values()))
        plt.xlabel("Number of cluster")
        plt.ylabel("Sum Squared Error")
        plt.show()

def run_agglom_sk(data_frame, col_1, col_2):
    
    X = data_frame.iloc[:, [col_1, col_2]].values
    agglom_cluster = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    print("==========================================")
    print("Result SK LEAR AGGLOM")
    print("==========================================")
    print(agglom_cluster)

def runplot_aggolm_sp(data_frame, plot=False):

    var_row = data_frame.index
    row_dist_df = pd.DataFrame(squareform(
        pdist(data_frame, metric='euclidean')), columns=var_row, index=var_row)

    row_distdf_r1 = linkage(
        row_dist_df.values, method='complete', metric='euclidean')

    print("==========================================")
    print("agglomeration complete")
    print("For Graphs change plot = True")
    print("Analysis in report")
    print("==========================================")

    if plot ==True:
        dendr_plot = dendrogram(row_distdf_r1)
        plt.tight_layout
        plt.ylabel('Euclidean Distance')
        plt.show()


def runplot_DBSCAN(data_frame, col_1, col_2,plot=False):
    X = data_frame.iloc[:, [col_1, col_2]].values
    K = len(X) - 1
    n = len(X)

    db_clusters = DBSCAN(eps=10, min_samples=10, metric='euclidean')
    clusters = db_clusters.fit_predict(X)

    #get distances
    nearest_n = NearestNeighbors(n_neighbors=n)
    nbrs = nearest_n.fit(X)
    dist, index = nbrs.kneighbors(X)

    db_clusters = DBSCAN(eps=1, min_samples=20, metric='euclidean')
    clusters = db_clusters.fit_predict(X)
    print("==========================================")
    print("cluster results")
    print("I explain how to get optimal clusters in report")
    print("but I did not intergrate it into this algorithm")
    print(clusters)
    print("==========================================")

    #Kdist graph (sorted)
    distance_per_i = np.empty([K, n])
    for i in range(K):
        distance_ki = dist[:, (i+1)]
        distance_ki.sort()
        #Reverse
        distance_ki = distance_ki[::-1]
        #assign reversed matrix to index of dist_per_i
        distance_per_i[i] = distance_ki
        #print(distance_ki)
    #print(distance_per_i[0])
    if plot == True:
        for i in range(K):
            plt.plot(distance_per_i[i], label='K=%d' % (i+1))
        plt.ylabel('distances')
        plt.xlabel('points')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.show()
    

def main():
    #load datasets
    df_iris = pd.read_csv('iris.csv')
    df_plates = pd.read_csv('faults.csv')

    #some preprocessing
    iris_aglom_df = df_iris
    iris_aglom_df['enum_vals'] = iris_aglom_df['class']
    iris_aglom_df['enum_vals'] = pd.factorize(iris_aglom_df['class'])[0]
    iris_aglom_df = iris_aglom_df.drop('class', axis=1)
    df_iris = df_iris.drop('enum_vals', axis=1)
    iris_clean_df = iris_aglom_df

    #lists for plotting
    one_to_10 = list(range(1,11))
    custom = list(range(1,21,2))

    #run algorithms

    runplot_kmeans(df_iris, col_1=1, col_2=2, run_list=one_to_10, plot=False)
    runplot_kmeans(df_plates, col_1=5, col_2=6, run_list=one_to_10, plot=True)

    runplot_aggolm_sp(iris_aglom_df, plot=False)
    runplot_aggolm_sp(df_plates,plot=False)

    run_agglom_sk(df_iris, col_1=2, col_2=3)
    run_agglom_sk(df_plates, col_1=2, col_2=3)

    runplot_DBSCAN(iris_clean_df, col_1=2, col_2=3, plot=False)
    runplot_DBSCAN(df_plates, col_1=2, col_2=3, plot=False)

if __name__ == '__main__':
    main()
