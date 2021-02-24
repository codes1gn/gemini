import tensorflow as tf

def conv2d(input, weight):
    return tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME')

