'''
:Name: Ryan C. Adams
:Course: CS487_Applied_Machine_Learning
:Implementation: Stochastic gradient descent
Adaline methond is interested in defining and minimizing continious cost functions.
Cost function to minimized is Sum Squared Errors (SSE)
'''
import pandas as pd
import numpy as np
import collections


class SGD(object):
    def __init__(self, eta, iters):
        '''
        simple constructor function
        :param: _iter - int, number of iterations
        :param: eta - int, acts as a scalar how much to change weights by
        '''
        self.eta = eta
        self.iters = iters


    def shuffle(self, row_vectors, output_vectors):
        self.permutations = np.random.permutation(len(output_vectors))
        return row_vectors[self.permutations], output_vectors[self.permutations]


    def learn(self, row_vectors, output_vectors):
        '''

        '''
        #Generate random number for the length of all rows
        generator = np.random.RandomState(1)
        #initialize weigths function
        self.weights = generator.normal(loc=0.0, scale=.001, size=1 + row_vectors.shape[1])
        self.avg_costs = []
        self.errors_list = []
        for iter in range(self.iters):
            error = 0
            #shuffle step, creates permutted list of options
            row_vectors, output_vectors = self.shuffle(row_vectors, output_vectors)
            self.cost = []
            for row_vector, output_vector in zip(row_vectors,output_vectors):
                self.cost.append(self.update_weights(row_vector, output_vector))
            avg_cost = sum(self.cost) / len(output_vectors)
            self.avg_costs.append(avg_cost)
            for row_vector, output_vector in zip(row_vectors, output_vectors):
                prediction = self.predict(row_vector)
                error = error + np.where(output_vector == prediction, 0, 1)
            self.errors_list.append(error)
        return self
            

    def update_weights(self, row_vector, output_vector):
        '''
        Similar to the adaline method excpt it is going row by row and updating
        and is just taking the eta * error instead of summing using error.sum
        param: row_vector: one row, from the list of rows
        param: output_vector: one output from the outputs
        returns: cost: 
        '''
        #create a prediction using the weights and given row_vector
        output = np.dot(row_vector, self.weights[1:]) + self.weights[0]
        error = (output_vector - output)
        self.weights[1:] = self.weights[1:] + (self.eta * row_vector.T.dot(error))
        self.weights[0] = self.weights[0] + (self.eta * error)
        cost = (error**2) / 2
        return cost


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

