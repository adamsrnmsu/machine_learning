'''
:Name: Ryan C. Adams
:Course: CS487_Applied_Machine_Learning
:Implementation: Perceptron
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


class Perceptron(object):
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
        Moves through each row of attributes, and finds a prediction.
        Then if necessary, updates the weights to see if 
        '''
        #Generate random number for the length of all rows
        generator = np.random.RandomState(1)

        #Because the output_vector and row_vector sizes are equal we just pick one to find size
        self.weights = generator.normal(loc=0.0, scale=.01, size=len(row_vectors[0])+1)
        for iter in range(self.iters):
            error = 0  # initializes error counter to be zero of iter
            i = 0
            for row_vector, output_vector in zip(row_vectors, output_vectors):

                #create a prediction using the weights and given row_vector
                prediction = self.predict(row_vector)
                error = error + np.where(output_vector == prediction, 0, 1)
                self.weights[0] = self.weights[0] + self.eta * (output_vector - prediction)
                self.weights[1:] = self.weights[1:] + self.eta * (output_vector - prediction) * row_vector
                i = i+1
                #if i>=51:
                #    break
            print(f"error: {int(error)}, weights {self.weights}")


    def predict(self, row_vector):
        '''
        Takes a row_vector and uses dot product across weights, if output is positive
        scales the prediction to be 1, else zeros
        :param: row_vector - vector, that contains attribute data about single sample
        :returns: a prediction as a 1 or -1
        '''
        input = np.dot(row_vector, self.weights[1:]) + self.weights[0]
        prediction = np.where(input >= 0, 1, -1)
        return prediction

