import unittest
import ast
import astunparse
import time

from absl import flags

from gemini.compiler import *
from gemini.utils import *


class TestGeminiCompilerDump(unittest.TestCase):

    def setUp(self):
        # prepare codes, compilers
        self.filename = "./samples/mnist.py"
        self.compiler = GeminiCompiler()
        self.code_str = read_src(self.filename)
        self.compiler.parse(self.code_str, filename=self.filename)

    def tearDown(self):
        del self.compiler
        del self.code_str
        del self.filename

    # method to test dump raw strings

    def test_dump(self):
        # TODO change to optional of python
        dump_src = self.compiler.dump(pretty=False)
        self.assertNotEqual(dump_src, '')

    def test_run_with_src(self):
        self.compiler.run(globals(), use_ast=False)


if __name__ == '__main__':
    unittest.main()
