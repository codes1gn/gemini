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

    @classmethod
    def dump(cls, pretty=False):
        # type: (Bool) -> None

        # do sanity check
        assert(cls._initialized)
        assert(cls._ast_root is not None)
        assert(isinstance(cls._ast_root, ast.AST))

        # dump with raw or formatted way
        if pretty:
            astunparse.dump(cls._ast_root)
        else:
            ast.dump(cls._ast_root)
        return

    # TODO set classmethdo
    # python2 not support typing hint, thus leave a TODO here

    def parse_function(self, func):
        # type: (Callable[..., Any]) -> None
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
        ast.increment_lineno(ast_root, n=start_lineno - 1)
        vlog('dump ast_root = ', ast.dump(ast_root))

        assert(isinstance(ast_root, ast.AST))
        self._ast_root = ast_root
        self._initialized = True
        return

        # # get module body
        # ast_fdef = ast_root.body[0]
        # vlog('dump ast_function_def = ', ast.dump(ast_fdef, include_attributes=True))

        # # TODO find out do what
        # import funcsigs
        # f_signature = funcsigs.signature(func)
        # f_params = f_signature.parameters
