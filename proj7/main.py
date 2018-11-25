""""
: Name: Ryan C. Adams
: Course: CS487 Applied Machine Learning I
: Project 7: Conv-NN
"""

import numpy as np
import tensorflow as tf
import os
import struct
import pprint as pp

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

def get_MNIST():
    mnist = fetch_mldata('MNIST original')
    X_mnist, y_mnist = mnist['data'], mnist['target']
    print("\n\n=======================================")
    print("retrieving MNIST")
    print(X_mnist.shape)
    print(y_mnist.shape)

    return X_mnist, y_mnist


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def tt_split(x_vector, y_vector):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        x_vector, y_vector, test_size=0.7, random_state=0)

    return Xtrain, Xtest, Ytrain, Ytest


def standardize_x(X_train, Xtest, Ytest):
    means= np.mean(X_train,axis=0)
    y_means = np.mean(Ytest, axis=0)
    std_val = np.std(X_train)
    X_train_centered = (X_train - means)/std_val
    Xtest_centered = (Xtest - means)/std_val
    Ytest_centered = (Ytest - y_means)/std_val

    return X_train_centered, Xtest_centered, Ytest_centered

def batch_generator(X, y, batch_size=100, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(seed=random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    for i in range(0, X.shape[0], batch_size):
        yield(X[i:i+batch_size,:], y[i:i+batch_size])

class ConvNN(object):
    def __init__(self, batchsize=100, epochs=20, learning_rate= .0001,
                dropout_rate=0.4, shuffle=True, random_seed=None):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle

        g = tf.Graph() #create emptygraph
        with g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self. init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        
        self.sess = tf. Session(graph=g)
    
    def build(self):
        tf_x = tf.placeholder(tf.float32, shape = [None, 784], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name = 'tf_y')
        is_train = tf.placeholder(tf.bool, shape=(), name= 'is_train') #for dropout

        ## reshape x to a 4d tensor: [batchsize, width, height, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name = 'input_x_2dimages')
        ## One-hot ecoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name='input_y_onehot')

        ## 1st layer: conv
        h1 = tf.layers.conv2d(tf_x_image, kernel_size=(3,3), strides = (1,1), filters=4, padding = 'valid', activation=tf.nn.relu)
        h1_pool = tf.layers.max_pooling2d(h1, pool_size=(2,2), strides=(2,2))

        print("h1 shape:\t ", tf.shape(h1))
        print("h1_pool :\t", tf.shape(h1_pool))

        ## 2n layer: Conv_2
        h2 = tf.layers.conv2d(h1_pool, kernel_size=(3, 3), strides=(
            3, 3), filters=2, activation=tf.nn.relu, padding='valid')
        h2_pool = tf.layers.max_pooling2d(h2, pool_size=(4,4), strides=(4,4))

        print("h2 shape:\t", tf.shape(h2))
        print("h2_pool :\t", tf.shape(h2_pool))

        ## 3rd layer : Fully Connected
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, shape=[-1,n_input_units])
        h3 = tf.layers.dense(h2_pool_flat, 10, activation=tf.nn.relu)

        print("h3 shape\t", tf.shape(h3))

        # ##Dropout
        # h3_drop = tf.layers.dropout(h3, rate= self.dropout_rate, training=is_train)

        # ## 4th layer: Fully Connected (linear activation)
        # h4 = tf.layers.dense(h3_drop, 10, activation=None)
        # print(h4)

        ## Prediction
        predictions = {
            'probabilities': tf.nn.softmax(h3, name='probabilities'),
            'labels': tf.cast(tf.argmax(h3, axis=1), tf.int32,
                              name='labels')
        }
        
        ## Loss Function and Optimization

        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=h3, labels=tf_y_onehot), name='cross_entropy_loss')
        
        ## optimizer:
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

        ## Finding accuracy
        correct_predictions = tf.equal(
            predictions['labels'],
            tf_y, name='correct_preds')

        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy')

        print("Build Complete")

    def train(self, training_set, validation_set=None, initialize=True):
        ## initialize variables

        if initialize:
            self.sess.run(self.init_op)

        self.train_cost_= []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])

        for epoch in range(1, self.epochs +1):
            batch_gen = batch_generator(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'is_train:0': True}
                loss, _ = self.sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
                avg_loss += loss
        
        print('Epoch %02d: Training Avg. Loss: ''%7.3f' % (epoch, avg_loss), end = ' ')
        if validation_set is not None:
            feed = {'tf_x:0': batch_x,
                    'tf_y:0': batch_y, 'is_train:0': False}
            valid_acc = self.sess.run('accuracy:0', feed_dict=feed)
            print('Validation Acc: %7.3f' % valid_acc)
        else:
                print()
        
        print("Trining Complete no errors")
        
    def predict(self, X_test, return_proba = False):
        feed = {'tf_x:0', X_test, 'is_train:0', False}
        if return_proba:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0', feed_dict=feed)
    
    def save(self, epoch, path='./tflayers-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model in %s' % path)
        self.saver.save(self.sess, os.path.join(path, 'model.ckpt'), global_step=epoch)
    
    def load(self, epoch, path):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess, os.path.join(path, 'model.ckpt-%d' % epoch))


def main():
    #get data
    #X_mnist, y_mnist = get_MNIST()

    X_data, y_data = load_mnist('./', kind='train')

    print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))
    X_test, y_test = load_mnist('./', kind='t10k')
    print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))

    X_train, y_train = X_data[:50000, :], y_data[:50000]
    X_valid, y_valid = X_data[50000:, :], y_data[50000:]

    print('Training:   ', X_train.shape, y_train.shape)
    print('Validation: ', X_valid.shape, y_valid.shape)
    print('Test Set:   ', X_test.shape, y_test.shape)

    mean_vals = np.mean(X_train, axis=0)
    std_val = np.std(X_train)

    X_train_centered = (X_train - mean_vals)/std_val
    X_valid_centered = (X_valid - mean_vals)/std_val
    X_test_centered = (X_test - mean_vals)/std_val

    # #split data
    # Xtrain, Xtest, Ytrain, Ytest = tt_split(X_mnist, y_mnist)

    # normalize data
    # X_train_centered, Xtest_centered, Ytest_centered = standardize_x(
    #     Xtrain, Xtest, Ytest)


    #Create CNN model and delete it

    epochs = [1,2,5,10,20]

    end_dict = {}
    
    for epoch in epochs:
        start = timer()
        cnn = ConvNN(random_seed=123,epochs=epoch)
        cnn.train(training_set=(X_train_centered, y_train),
              validation_set=(X_valid_centered, y_valid))
        end = timer()
        end_dict[epoch] = end-start
        
    print("Timings")
    print("Epoch, Time in seconds")
    pp.pprint(end_dict)

    # cnn.save(epoch=20)
    # del cnn

    # cnn2 = ConvNN(random_seed=123)
    # cnn2.load(epoch=20, path='./tflayers-model/')
    # print(cnn2.predict(Xtest_centered[:10,:]))

    # preds = cnn.predict(X_test_centered)
    # print('Test Accuracy: %.2f%%' % (100*np.sum(y_test == preds)/len(y_test)))

if __name__ == '__main__':
    main()
