import unittest
import ast
import time

from gemini.compiler import *
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
    sub_1 = input1 - input2
    mul_1 = tf.multiply(add_1, sub_1)
    return mul_1
        """

        self.func = model
        self.compiler = GeminiCompiler()

        pass

    def tearDown(self):
        del self.compiler
        del self.func
        del self.code_str

    # method to test dump raw strings
    def test_raw_dump_by_func(self):
        self.compiler.parse_function(self.func)
        # TODO change to optional of python
        while not self.compiler.inited:
            time.sleep(10)
        dump_str = ast.dump(self.compiler.ast)
        print(dump_str)
        pass


if __name__ == '__main__':
    unittest.main()
