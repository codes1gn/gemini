import ssl
from mnist_lib import conv2d
import tensorflow as tf
import os
import numpy as np
import sys
import time
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data
import enflame_cake_cutter as ecc
print('hello world')
print(os.path.abspath(__file__))
print(os.environ.get("PYTHONPATH"))

ssl._create_default_https_context = ssl._create_unverified_context

tf.app.flags.DEFINE_string("device", "dtu", "cpu|gpu|dtu|xla_cpu|xla_gpu")
tf.app.flags.DEFINE_integer("max_step", 10, "max training steps")
tf.app.flags.DEFINE_integer(
    "display_step",
    1,
    "display loss every several steps")
tf.app.flags.DEFINE_boolean(
    'display_loss',
    True,
    'whether to show loss and train accuracy')
tf.app.flags.DEFINE_integer(
    'batch_size',
    64,
    """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, "learning rate")
tf.app.flags.DEFINE_string("dtype", "fp32", "fp32|bf16|fp16")

FLAGS = tf.app.flags.FLAGS

device_map = {
    'cpu': "/cpu:0",
    'gpu': "/gpu:0",
    'dtu': "/device:XLA_DTU:0",
    'xla_cpu': "/device:XLA_CPU:0",
    'xla_gpu': "/device:XLA_GPU:0",
}
device_name = device_map[FLAGS.device]
if FLAGS.device in ['dtu', 'xla_cpu', 'xla_gpu']:
    use_resource = True
else:
    use_resource = False

dtype_map = {
    'fp32': tf.float32,
    'bf16': tf.bfloat16,
    'fp16': tf.half
}
training_type = dtype_map[FLAGS.dtype]

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


def weight_variable(shape, name, dtype):
    with tf.device('/cpu:0'):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=dtype)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name, dtype):
    initial = tf.constant(0, shape=shape, dtype=dtype)
    return tf.Variable(initial, name=name)


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def mnist_model(x, keep_prob, dtype=tf.float32):
    """
    network:
        conv 5X5X32 + relu
        max pooling 2X2
        conv 5X5X32 + relu
        max pooling 2X2
        fully connected 256 units + relu (with dropuot)
        fully connected 10 units + softmax (with dropuot)
    params:
        x: input data
        keep_prob: dropout radio
    return:
        logits: output of model
    """
    with tf.device('/device:XLA_DTU:3'):
        with tf.name_scope("Reshaping_image") as scope:
            x_image = tf.cast(x, dtype)
            x_image = tf.reshape(x_image, [-1, 28, 28, 1])

    with tf.device('/device:XLA_DTU:17'):
        with tf.name_scope("Conv1") as scope:
            w_conv1 = weight_variable(
                [5, 5, 1, 32], dtype=dtype, name='weight_for_Conv_Layer_1')
            b_conv1 = bias_variable(
                [32], dtype=dtype, name='bias_for_Conv_Layer_1')
            h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

    with tf.device('/device:XLA_DTU:8'):
        with tf.name_scope("Conv2") as scope:
            W_conv2 = weight_variable(
                [5, 5, 32, 64], dtype=dtype, name='weight_for_Conv_Layer_2')
            b_conv2 = bias_variable(
                [64], dtype=dtype, name='bias_for_Conv_Layer_2')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

    with tf.device('/device:XLA_DTU:13'):
        with tf.name_scope("Fully_Connected1") as scope:
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            W_fc1 = weight_variable(
                [7 * 7 * 64, 1024], dtype=dtype, name='weight_for_Fully_Connected_layer_1')
            b_fc1 = bias_variable(
                [1024],
                dtype=dtype,
                name='bias_for_Fully_Connected_Layer_1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.device('/device:XLA_DTU:32'):
        with tf.name_scope("Fully_Connected2") as scope:
            W_fc2 = weight_variable(
                [1024, 10], dtype=dtype, name='weight_for_Fully_Connected_layer_2')
            b_fc2 = bias_variable(
                [10], dtype=dtype, name='bias_for_Fully_Connected_Layer_2')
            net = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return net


def create_config():
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True,
                            device_count={'CPU': 8},
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)
    ## config.dtu_options.visible_device_list = str(local_rank)
    # print("albert config.dtu_options.visible_device_list={}".format(config.dtu_options.visible_device_list))
    from tensorflow.core.protobuf import rewriter_config_pb2
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = off
    return config


def mnist_test():
    mnist = input_data.read_data_sets(
        train_dir="{}/dataset".format(ROOT_PATH), one_hot=True)
    x = tf.placeholder(training_type, shape=[None, 784])
    y_ = tf.placeholder(training_type, shape=[None, 10])
    keep_prob = tf.placeholder(training_type, shape=[])
    global_step = tf.train.get_or_create_global_step()

    total_dtu_duration = 0.0

    def build_net():
        # with tf.device(device_name):
        with tf.device('/device:CPU:0'):
            with tf.variable_scope('', dtype=training_type, use_resource=use_resource):
                logits = mnist_model(x, keep_prob, dtype=training_type)
                logits = tf.cast(logits, tf.float32)

        # with tf.device('/device:CPU:0'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)

        train_op = tf.train.AdamOptimizer(
            FLAGS.learning_rate).minimize(
            cross_entropy, global_step)
        # train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, use_resource=use_resource).minimize(cross_entropy, global_step)

        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(
                        tf.nn.softmax(logits), 1), tf.argmax(
                        y_, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return cross_entropy, train_op, accuracy

    with tf.Session(config=create_config()) as sess:
        _cross_entropy, _train_op, _accuracy = build_net()
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(1, FLAGS.max_step + 1):
            train_x, train_y = mnist.train.next_batch(FLAGS.batch_size)

            step_start = time.time()
            _, loss = sess.run([_train_op, _cross_entropy], feed_dict={
                               x: train_x, y_: train_y, keep_prob: 0.5})
            step_duration = time.time() - step_start
            total_dtu_duration += step_duration

            if i % FLAGS.display_step == 0 and FLAGS.display_loss:
                summary, train_accuracy, step = sess.run([merged, _accuracy, global_step],
                                                         feed_dict={x: train_x,
                                                                    y_: train_y,
                                                                    keep_prob: 1.0})
                print("{}, global_step={}, train_accuracy={}, loss={}, step_duration={}".format(
                    datetime.now().strftime('%Y-%m-%d,%H:%M:%S'), step, train_accuracy, loss, step_duration))
        print("total_dtu_duration: {}s".format(total_dtu_duration))


if __name__ == '__main__':
    mnist_test()
