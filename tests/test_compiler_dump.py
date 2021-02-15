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

    def tearDown(self):
        del self.compiler
        del self.func
        del self.code_str

    # method to test dump raw strings
    def test_raw_dump_by_func(self):
        self.compiler.parse_function(self.func)
        # TODO change to optional of python
        dump_src = self.compiler.dump(pretty=False)
        expected_src = "Module(body=[FunctionDef(name='model', args=arguments(args=[Name(id='input1', ctx=Param()), Name(id='input2', ctx=Param())], vararg=None, kwarg=None, defaults=[]), body=[Assign(targets=[Name(id='add_1', ctx=Store())], value=Call(func=Attribute(value=Name(id='tf', ctx=Load()), attr='add', ctx=Load()), args=[Name(id='input1', ctx=Load()), Name(id='input2', ctx=Load())], keywords=[], starargs=None, kwargs=None)), Assign(targets=[Name(id='sub_1', ctx=Store())], value=BinOp(left=Name(id='input1', ctx=Load()), op=Sub(), right=Name(id='input2', ctx=Load()))), Assign(targets=[Name(id='mul_1', ctx=Store())], value=Call(func=Attribute(value=Name(id='tf', ctx=Load()), attr='multiply', ctx=Load()), args=[Name(id='add_1', ctx=Load()), Name(id='sub_1', ctx=Load())], keywords=[], starargs=None, kwargs=None)), Return(value=Name(id='mul_1', ctx=Load()))], decorator_list=[])])"
        self.assertEqual(expected_src, dump_src)

    # method to test dump raw strings
    def test_pretty_dump_by_func(self):
        self.compiler.parse_function(self.func)
        # TODO change to optional of python
        dump_src = self.compiler.dump(pretty=True)
        expected_src = """Module(body=[FunctionDef(
  name='model',
  args=arguments(
    args=[
      Name(
        id='input1',
        ctx=Param()),
      Name(
        id='input2',
        ctx=Param())],
    vararg=None,
    kwarg=None,
    defaults=[]),
  body=[
    Assign(
      targets=[Name(
        id='add_1',
        ctx=Store())],
      value=Call(
        func=Attribute(
          value=Name(
            id='tf',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[
          Name(
            id='input1',
            ctx=Load()),
          Name(
            id='input2',
            ctx=Load())],
        keywords=[],
        starargs=None,
        kwargs=None)),
    Assign(
      targets=[Name(
        id='sub_1',
        ctx=Store())],
      value=BinOp(
        left=Name(
          id='input1',
          ctx=Load()),
        op=Sub(),
        right=Name(
          id='input2',
          ctx=Load()))),
    Assign(
      targets=[Name(
        id='mul_1',
        ctx=Store())],
      value=Call(
        func=Attribute(
          value=Name(
            id='tf',
            ctx=Load()),
          attr='multiply',
          ctx=Load()),
        args=[
          Name(
            id='add_1',
            ctx=Load()),
          Name(
            id='sub_1',
            ctx=Load())],
        keywords=[],
        starargs=None,
        kwargs=None)),
    Return(value=Name(
      id='mul_1',
      ctx=Load()))],
  decorator_list=[])])"""
        self.assertEqual(expected_src, dump_src)


if __name__ == '__main__':
    unittest.main()
