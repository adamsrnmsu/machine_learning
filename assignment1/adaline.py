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
        self.weights = generator.normal(loc=0.0, scale=.001, size= 1+ row_vectors.shape[1])
        self.costs = []
        for iter in range(self.iters):
            errors = 0
            #create a prediction using the weights and given row_vector
            output = np.dot(row_vectors, self.weights[1:]) + self.weights[0]
            errors = (output_vectors - output)
            self.weights[1:] = self.weights[1:] + (self.eta * row_vectors.T.dot(errors))
            self.weights[0] = self.weights[0] + (self.eta * errors.sum())
            self.costs.append((errors ** 2).sum() / 2.0)
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


def main():
    html = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    attributes = ['sepal_length', 'sepal_width',
                  'petal_length', 'petal_width', 'class']
    df = construct_pandas_frame(html, attributes)
    x = df.iloc[0:100, [0, 2]].values  # row vector
    x_std = np.copy(x)
    #standardize the input features
    x_std[:, 0] =  (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    x_std[:, 1] =  (x[:, 0] - x[:, 1].mean()) / x[:, 1].std()
    # output vectors of size 1, equal to target classification
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)

    # This is just a quick check we have 50 setosa, 50 not setosa
    verify_count = collections.Counter(y)
    print(f"should be 50 ==1 and 50 == -1: {verify_count}")

    #create / train perceptron
    model = Adaline(eta=.1, iters=14)
    model.learn(x_std, y)
    for cost in enumerate(model.costs,0):
        print(cost)


if __name__ == '__main__':
    main()
