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
        '_initialized',
        '_src_file',
    ]

    def __init__(self):
        self._ast_root = None
        self._source_code = None
        self._initialized = False
        self._src_file = None

    @property
    def ast(self):
        return self._ast_root

    @property
    def src(self):
        try:
            self._source_code = astunparse.unparse(self._ast_root)
        except Exception:
            assert 0, 'unparse failed'
        return self._source_code

    @property
    def inited(self):
        return self._initialized

    @property
    def src_file(self):
        return self._src_file

    # TODO add pretty dump

    def _apply_postprocess_transformer(self, transformer):
        if isinstance(transformer, SetParentTransformer):
            return
        if isinstance(transformer, ShardingLeastDimTransformer):
            postproc_transformer = ShardingLeastDimPostTransformer(transformer)
            self._ast_root = postproc_transformer.visit(self._ast_root)
            # TODO(albert) remember to add exception handles
            return

    def apply_transformer(self, transformer):
        # type: (BaseTransformer) -> None
        assert self._initialized, "compiler not inited"
        assert isinstance(
            transformer, BaseTransformer), "given arg is not of type BaseTransformer"
        self._ast_root = transformer.visit(self._ast_root)
        self._apply_postprocess_transformer(transformer)
        return

    def run(self, environment, use_ast=False):
        # print('global keys have\n')
        # print(globals().keys())
        # TODO defaultly, use src to run, rather than ast
        assert self._initialized, "compiler not inited"
        if use_ast:
            exec(
                compile(
                    self._ast_root,
                    filename=self._src_file,
                    mode='exec'),
                environment)
        else:
            exec(self._source_code, environment)
        pass

    def dump(self, pretty=True, dump_file=""):
        # type: (Bool) -> str

        # do sanity check
        assert self._initialized, "compiler not inited"
        assert self._ast_root is not None, "compiler.ast is None"
        assert isinstance(
            self._ast_root, ast.AST), "compiler.ast is not of type ast.AST"

        # dump with raw or formatted way
        if pretty:
            return astunparse.dump(self._ast_root)
        else:
            return ast.dump(self._ast_root)

    # TODO set classmethod
    # python2 not support typing hint, thus leave a TODO here

    # TODO legacy codes for setting parents
    # def _set_parents(self):
    #     # this function sets parents of ast tree, that add a way to access parent nodes for pass
    #     for stmt in ast.walk(self._ast_root):
    #         print(stmt)
    #         if stmt is None:
    #             print('super')
    #         for child in ast.iter_child_nodes(stmt):
    #             print(child)
    #             # setattr(child, 'gemini_parent', stmt)
    #     return

    def parse(self, func_or_src, filename="dummy.py"):
        # type: (Callable[..., Any]) -> None
        assert isinstance(func_or_src, Callable) or isinstance(
            func_or_src,
            basestring), "object to parse is not in type Callable or source code string"
        # for python3.x, do assert(isinstance(func, Callable) or
        # isinstance(func, str))

        if isinstance(func_or_src, Callable):
            func = func_or_src
            src_filename = inspect.getsourcefile(func)
            self._src_file = src_filename
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
            src_filename = filename
            self._src_file = src_filename
            src_code = func_or_src
            ast_root = ast.parse(src_code, filename=src_filename)
            # ast.increment_lineno(ast_root, n=0)

        assert isinstance(
            ast_root, ast.AST), "compiler.ast is not of type ast.AST"
        # TODO set parents of each node for further operations
        # TODO leave a tree arg here tmp
        # self._set_parents()
        # assert 0
        self._ast_root = ast_root
        # TODO(albert) remove this assign, and move assign to property, keep
        # ast as master
        self._source_code = src_code
        self._initialized = True
        return

        # # get module body
        # ast_fdef = ast_root.body[0]
        # vlog('dump ast_function_def = ', ast.dump(ast_fdef, include_attributes=True))

        # # TODO find out do what
        # import funcsigs
        # f_signature = funcsigs.signature(func)
        # f_params = f_signature.parameters
