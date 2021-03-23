import unittest
import tensorflow as tf
import os
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class SinglePointToPointTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._use_resource = True
        cls._dtype_map = {
            'fp32': tf.float32,
            'bf16': tf.bfloat16,
            'fp16': tf.half
        }
        cls._training_type = tf.float32
        cls._ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
        cls._seed = 1234

        def create_config():
            config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True,
                                    device_count={'CPU': 8},
                                    inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1)
            # config.dtu_options.visible_device_list = str(local_rank)
            print(
                "albert config.dtu_options.visible_device_list={}".format(
                    config.dtu_options.visible_device_list))
            from tensorflow.core.protobuf import rewriter_config_pb2
            off = rewriter_config_pb2.RewriterConfig.OFF
            config.graph_options.rewrite_options.memory_optimization = off
            return config
        cls._config = create_config()
        cls._initer = tf.compat.v1.random_uniform_initializer

    @classmethod
    def tearDownClass(cls):
        del cls._dtype_map
        del cls._config

    def test_single_point_to_point_0(self):
        def model(devices):
            with tf.device(devices[0]):
                x = tf.get_variable(
                    'x1',
                    shape=[
                        1107,
                        1301],
                    dtype=tf.float32,
                    initializer=self._initer(
                        self._seed))
                x = tf.multiply(x, x)
            with tf.device(devices[1]):
                x = x * 0.8
                x = tf.nn.relu(x)
                output = x
            return output

        with tf.Session(config=self._config) as sess:
            with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
                model_cross_card = model(
                    ['/device:XLA_DTU:0', '/device:XLA_DTU:6'])
                model_cross_cluster = model(
                    ['/device:XLA_DTU:2', '/device:XLA_DTU:3'])
                model_ref = model(['/CPU:0', '/CPU:1'])

                init = tf.global_variables_initializer()
                sess.run(init)

                _result_cross_card = sess.run([model_cross_card])
                _result_cross_cluster = sess.run([model_cross_cluster])
                _result_ref = sess.run([model_ref])
                self.assertEqual(
                    np.array(_result_ref).all(),
                    np.array(_result_cross_cluster).all())
                self.assertEqual(
                    np.array(_result_ref).all(),
                    np.array(_result_cross_card).all())

    def test_single_point_to_point_1(self):
        def model(devices):
            with tf.device(devices[0]):
                x = tf.get_variable(
                    'x2',
                    shape=[
                        323,
                        774],
                    dtype=tf.float32,
                    initializer=self._initer(
                        self._seed))
                x = tf.multiply(x, x)
            with tf.device(devices[1]):
                x = x + x - 2.0
                x = tf.nn.relu(x)
                output = x
            return output

        with tf.Session(config=self._config) as sess:
            with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
                model_cross_card = model(
                    ['/device:XLA_DTU:24', '/device:XLA_DTU:36'])
                model_cross_cluster = model(
                    ['/device:XLA_DTU:20', '/device:XLA_DTU:17'])
                model_ref = model(['/CPU:0', '/CPU:1'])

                init = tf.global_variables_initializer()
                sess.run(init)

                _result_cross_card = sess.run([model_cross_card])
                _result_cross_cluster = sess.run([model_cross_cluster])
                _result_ref = sess.run([model_ref])
                self.assertEqual(
                    np.array(_result_ref).all(),
                    np.array(_result_cross_cluster).all())
                self.assertEqual(
                    np.array(_result_ref).all(),
                    np.array(_result_cross_card).all())

    def test_single_point_to_point_2(self):
        def model(devices):
            with tf.device(devices[0]):
                x = tf.get_variable(
                    'x3',
                    shape=[
                        235,
                        174],
                    dtype=tf.float32,
                    initializer=self._initer(
                        self._seed))
                x = tf.multiply(x, x)
            with tf.device(devices[1]):
                x = x + x - 2.0
                x = tf.nn.relu(x)
                output = x
            return output

        with tf.Session(config=self._config) as sess:
            with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
                model_cross_card = model(
                    ['/device:XLA_DTU:10', '/device:XLA_DTU:15'])
                model_cross_cluster = model(
                    ['/device:XLA_DTU:10', '/device:XLA_DTU:8'])
                model_ref = model(['/CPU:0', '/CPU:1'])

                init = tf.global_variables_initializer()
                sess.run(init)

                _result_cross_card = sess.run([model_cross_card])
                _result_cross_cluster = sess.run([model_cross_cluster])
                _result_ref = sess.run([model_ref])
                self.assertEqual(
                    np.array(_result_ref).all(),
                    np.array(_result_cross_cluster).all())
                self.assertEqual(
                    np.array(_result_ref).all(),
                    np.array(_result_cross_card).all())


if __name__ == '__main__':
    unittest.main()
