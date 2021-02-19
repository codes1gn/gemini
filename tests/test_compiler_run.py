import unittest
import ast
import astunparse
import time
import numpy as np

import tensorflow.compat.v1 as tf

from gemini.gemini_compiler import *
from gemini.utils import *


def model(input1, input2):
    add_1 = tf.add(input1, input2)
    sub_1 = input1 - input2
    mul_1 = tf.multiply(add_1, sub_1)
    return mul_1


class TestGeminiCompilerDump(unittest.TestCase):

    def setUp(self):
        # prepare codes, compilers
        self.code_str = """

def model(input1, input2):
    add_1 = tf.add(input1, input2)
    sub_1 = (input1 - input2)
    mul_1 = tf.multiply(add_1, sub_1)
    return mul_1
"""
        self.func = model
        self.compiler = GeminiCompiler()

    def tearDown(self):
        del self.compiler
        del self.func
        del self.code_str

    def test_run_with_src(self):
        self.compiler.parse(self.code_str)
        # TODO change to optional of python
        self.compiler.run(globals(), use_ast=False)
        _a = tf.constant(4, shape=[3, 3])
        _b = tf.constant(5, shape=[3, 3])
        logits = model(_a, _b)
        with tf.Session() as sess:
            _ = sess.run(logits)
            array1 = _
            array2 = np.array([
                [-9, -9, -9],
                [-9, -9, -9],
                [-9, -9, -9]
            ])
            self.assertTrue(np.array_equal(array1, array2))

    def test_run_with_ast(self):
        self.compiler.parse(self.code_str)
        # TODO change to optional of python
        self.compiler.run(globals(), use_ast=True)
        _a = tf.constant(1, shape=[3, 3])
        _b = tf.constant(2, shape=[3, 3])
        logits = model(_a, _b)
        with tf.Session() as sess:
            _ = sess.run(logits)
            array1 = _
            array2 = np.array([
                [-3, -3, -3],
                [-3, -3, -3],
                [-3, -3, -3]
            ])
            self.assertTrue(np.array_equal(array1, array2))


if __name__ == '__main__':
    unittest.main()
