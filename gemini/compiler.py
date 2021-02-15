import os
import inspect
import textwrap
import ast
import astunparse

from typing import Callable

from .utils import *
from .transformer import *

__all__ = [
    'GeminiCompiler',
]


class GeminiCompiler:

    __slots__ = [
        '_ast_root',
        '_source_code',
        '_initialized'
    ]

    def __init__(self):
        self._ast_root = None
        self._source_code = None
        self._initialized = False

    @property
    def ast(self):
        return self._ast_root

    @property
    def src(self):
        return self._source_code

    @property
    def inited(self):
        return self._initialized

    # TODO add pretty dump

    def apply_transformer(self, transformer):
        # type: (BaseTransformer) -> None
        assert(isinstance(transformer, BaseTransformer))
        self._ast_root = transformer.visit(self._ast_root)
        return

    def dump(self, pretty=True):
        # type: (Bool) -> str

        # do sanity check
        # assert(self._initialized)
        # assert(self._ast_root is not None)
        # assert(isinstance(self._ast_root, ast.AST))

        # dump with raw or formatted way
        if pretty:
            return astunparse.dump(self._ast_root)
        else:
            return ast.dump(self._ast_root)

    # TODO set classmethod
    # python2 not support typing hint, thus leave a TODO here

    def parse(self, func_or_src):
        # type: (Callable[..., Any]) -> None
        assert(
            isinstance(
                func_or_src,
                Callable) or isinstance(
                func_or_src,
                basestring))
        # for python3.x, do assert(isinstance(func, Callable) or
        # isinstance(func, str))

        if isinstance(func_or_src, Callable):
            func = func_or_src
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
            ast.increment_lineno(ast_root, n=start_lineno - 1)
        elif isinstance(func_or_src, basestring):
            src_filename = "dummy.py"
            src_code = func_or_src
            ast_root = ast.parse(src_code, filename=src_filename)
            ast.increment_lineno(ast_root, n=0)

        assert(isinstance(ast_root, ast.AST))
        self._ast_root = ast_root
        self._initialized = True
        vlog('dump ast_root = ', self.dump())
        return

        # # get module body
        # ast_fdef = ast_root.body[0]
        # vlog('dump ast_function_def = ', ast.dump(ast_fdef, include_attributes=True))

        # # TODO find out do what
        # import funcsigs
        # f_signature = funcsigs.signature(func)
        # f_params = f_signature.parameters
