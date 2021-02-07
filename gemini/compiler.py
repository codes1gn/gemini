import ast
import inspect
import textwrap
from .utils import *
import os
from typing import Callable

__all__ = [
    'GeminiCompiler',
           ]


class GeminiCompiler:

  __slots__ = [
      '_py_module'
  ]

  def __init__(self):
    self._py_module = None

  @property
  def py_module(self):
    return self._py_module

  def parse_function(self, func):
    # python2 not support typing hint, thus leave a TODO here
    # type: (Callable[..., Any]) -> ast.AST
    assert(isinstance(func, Callable))

    src_filename = inspect.getsourcefile(func)
    vlog('src_filename = ', src_filename)

    src_lines, start_lineno = inspect.getsourcelines(func)
    vlog('src_lines = ', src_lines)
    vlog('start_lineno = ', start_lineno)

    src_code = "".join(src_lines)
    vlog('src_code with indent = ', src_code)
    src_code = textwrap.dedent(src_code)
    vlog('src_code without indent = ', src_code)

    ast_root = ast.parse(src_code, filename=src_filename)
    # adjust start_lineno starting from 0 as file header
    ast.increment_lineno(ast_root, n=start_lineno-1)
    vlog('dump ast_root = ', ast.dump(ast_root))

    assert(isinstance(ast_root, ast.AST))
    return ast_root

    # # get module body
    # ast_fdef = ast_root.body[0]
    # vlog('dump ast_function_def = ', ast.dump(ast_fdef, include_attributes=True))

    # # TODO find out do what
    # import funcsigs
    # f_signature = funcsigs.signature(func)
    # f_params = f_signature.parameters
