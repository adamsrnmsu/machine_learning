'''
:Name: Ryan C. Adams
:Course: CS487_Applied_Machine_Learning
:Implementation: adaline
Adaline methond is interested in defining and minimizing continious cost functions.
Cost function to minimized is Sum Squared Errors (SSE)
'''
import pandas as pd
import numpy as np
import collections


def construct_pandas_frame(html, attributes):
    '''
    Creats a pandas dataframe from a csv like data format from a csv
    Also assumes that the header is not in the csv representation nor the index name
    :param: html - string for the location of the csv like data on website
    :param: attributes - list of strings of the given data set in order
    :returns: pandas dataframe
    '''
    df = pd.read_csv(html, header=None)
    df.columns = attributes
    return df


class Adaline(object):
    def __init__(self, eta, iters):
        '''
        simple constructor function
        :param: _iter - int, number of iterations
        :param: eta - int, acts as a scalar how much to change weights by
        '''
        self.eta = eta
        self.iters = iters

    def learn(self, row_vectors, output_vectors):
        '''
        '''
        #Generate random number for the length of all rows
        generator = np.random.RandomState(1)
        #Because the output_vector and row_vector sizes are equal we just pick one to find size
        self.weights = generator.normal(loc=0.0, scale=.01, size=len(row_vectors[0])+1)
        self.costs = []
        for iter in range(self.iters):
            #create a prediction using the weights and given row_vector
            self.output = np.dot(row_vectors, self.weights[1:]) + self.weights[0]
            self.errors = (output_vectors - self.output)
            self.weights[1:] = self.weights[1:] + (self.eta * row_vectors.T.dot(self.errors))
            self.weights[0] = self.weights[0] + (self.eta * self.errors.sum())
            self.costs.append((self.errors ** 2).sum() / 2.0)
        return self


    def predict(self, row_vector):
        '''
        Takes a row_vector and uses dot product across weights, if output is positive
        scales the prediction to be 1, else zeros
        :param: row_vector - vector, that contains attribute data about single sample
        :returns: a prediction as a 1 or -1
        '''
        input = np.dot(row_vector, self.weights[1:]) + self.weights[0]
        prediction = np.where(input >= 0.0, 1, -1)
        return prediction
