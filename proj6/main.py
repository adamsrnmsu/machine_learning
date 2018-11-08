""""
: Name: Ryan C. Adams
: Course: CS487 Applied Machine Learning I
: Project 6: PCA

Usage of the following sklearn classifers on

: Datasets:
    i. Iris dataset:
       'iris.csv'

    ii. MNIST dataset:
        loaded in from sci kit learn

: Reductioners:
    i.PCA (Scikit)
    ii. Linerar Discriminant Analysis (Scikit)
    iii. Kernel PCA method (Scikit)
"""

import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA

def run_plot_pca(X_dat, Y_target, components, plot=False):
    
    plot_dict = {}

    for component in components:
        pca = PCA(n_components=component)
        pca_fit = pca.fit(X_dat)
        X_dat_transform = pca.transform(X_dat)
        clf = LogisticRegression(
            random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_dat_transform, Y_target)
        score = clf.score(X_dat_transform, Y_target)
        plot_dict[component] = score * 100

    pprint.pprint(plot_dict)

    if plot == True:
        plt.figure()
        plt.plot(list(plot_dict.keys()), list(plot_dict.values()))
        plt.xlabel("Number of Components")
        plt.ylabel("Accuracy %")
        plt.show()


def run_plot_lda(X_dat, Y_target, components, plot=False):

    plot_dict = {}

    for component in components:
        lda = LinearDiscriminantAnalysis(n_components=component)
        lda_fit = lda.fit(X_dat, Y_target)
        X_dat_transform = lda.transform(X_dat)
        clf = LogisticRegression(
            random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_dat_transform, Y_target)
        score = clf.score(X_dat_transform, Y_target)
        plot_dict[component] = score * 100

    pprint.pprint(plot_dict)

    if plot == True:
        plt.figure()
        plt.plot(list(plot_dict.keys()), list(plot_dict.values()))
        plt.xlabel("Number of Components")
        plt.ylabel("Accuracy %")
        plt.show()


def run_plot_kpca(X_dat, Y_target, components, kernels, plot=False):
    plot_dict = {}
    for kernel in kernels:
        val_list = []
        comp_list = []
        val_comp = []
        for component in components:
            kpca = KernelPCA(n_components=component, kernel=kernel)
            kpca_fit = kpca.fit(X_dat)
            X_dat_transform = kpca.transform(X_dat)
            clf = LogisticRegression(
                random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_dat_transform, Y_target)
            score = clf.score(X_dat_transform, Y_target)
            #cat_ker_n = str(kernel)+ str(component)
            score = score *100
            val_list.append(score)
            comp_list.append(component)
        val_comp.append(comp_list)
        val_comp.append(val_list)
        plot_dict[kernel] = val_comp
            #plot_dict[kernel] = score * 100
    pprint.pprint(plot_dict) 
    
    if plot == True:
        for kernel in plot_dict.keys():
            plt.figure()
            plt.title("Kernel PCA: " + str(kernel))
            plt.plot(plot_dict[kernel][0], plot_dict[kernel][1])
            plt.xlabel("Number of Components")
            plt.ylabel("Accuracy %")
            plt.show()

def main():
    #read in iris
    df_iris = pd.read_csv('iris.csv')
    df_iris['enum_vals'] = df_iris['class']
    df_iris['enum_vals'] = pd.factorize(df_iris['class'])[0]
    df_iris = df_iris.drop('class', axis=1)
    #print(df_iris)

    iris_data = df_iris.iloc[:,:4]
    iris_target = df_iris.iloc[:,4:5]

    #load digits create data and target
    digits = load_digits()
    dig_dat = digits.data
    dig_tar = digits.target

    components_iris = [1, 2, 3, 4]
    components_digits = [1, 2, 3, 4, 5, 10, 20, 30, 40]

    kernels = ["linear", "poly", "rbf", "sigmoid", "cosine"]

 
    run_plot_pca(dig_dat, dig_tar, components_digits, plot=False)
    run_plot_pca(iris_data, iris_target, components_iris, plot=False)
    
    run_plot_lda(dig_dat, dig_tar, components_digits, plot=False)
    run_plot_lda(iris_data, iris_target, components_iris, plot=False)

    run_plot_kpca(dig_dat, dig_tar, components_digits, kernels, plot=False)
    run_plot_kpca(iris_data, iris_target, components_iris, kernels, plot=False)
    
 
if __name__ == '__main__':
    main()
