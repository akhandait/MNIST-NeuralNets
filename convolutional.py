import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def create_placeholder(n_H0, n_W0, n_C0, n_y):

	X = tf.placeholder('tf.float32', shape = (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder('tf.float32', shape = (None, n_y))
    return X, Y

def initialize_parameters():

	W1 = 

