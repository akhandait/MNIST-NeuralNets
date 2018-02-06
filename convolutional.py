import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def create_placeholder(n_H0, n_W0, n_C0, n_y):

	X = tf.placeholder('tf.float32', shape = (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder('tf.float32', shape = (None, n_y))
    return X, Y

def initialize_parameters():

	W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer)
	W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer)
    
    parameters = {"W1" = W1, "W2 = W2"}
    return parameters

def forward_propagation(X, parameters):

	W1 = parameters["W1"]
	W2 = parameters["W2"]

	Z1 = tf.nn.conv2D(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    P2 = tf.contrib.layers.flatten(P2) 
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

    return Z3


    


