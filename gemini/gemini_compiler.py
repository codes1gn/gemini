import inspect
import textwrap
import ast
import astunparse
import importlib as impl

from typing import Callable

from gemini.pass_manager import *
from gemini.utils import *

__all__ = [
    'GeminiCompiler',
]


class GeminiCompiler:

    __slots__ = [
        '_ast_root',
        '_source_code',
        '_src_file',
        '_pass_manager',
        '_import_code_vector',
    ]

    def __init__(self):
        self._ast_root = None
        self._source_code = ""
        self._src_file = ""
        self._pass_manager = None
        self._import_code_vector = []

    @property
    def ast(self):
        return self._ast_root

    @property
    def src(self):
        try:
            self._source_code = astunparse.unparse(self._ast_root)
        except Exception:
            assert 0, 'unparse ast_root failed, cannot update source_code'
        return self._source_code

    @property
    def inited(self):
        return True if self.ast is not None else False

    @property
    def src_file(self):
        return self._src_file

    # method to apply MP patterns
    def apply_model_parallel(self, config):
        # TODO(albert) add config class
        if config['mode'] == "sharding":
            self._pass_manager = ShardingPassManager()
            self._pass_manager.register_passes()
            self._pass_manager.run(self)

    def compile_and_run(self, environment, use_ast=False):
        # print('global keys have\n')
        # print(globals().keys())
        # TODO defaultly, use src to run, rather than ast
        assert self.inited, "compiler not inited"
        if use_ast:
            assert isinstance(
                self._ast_root, ast.AST), "expected ast.AST, but got " + str(type(self._ast_root))
            co_obj = compile(
                self._ast_root,
                filename=self._src_file,
                mode='exec')
            exec(co_obj, environment)
        else:
            exec(self._source_code, environment)
        pass

    def raw_dump(self):
        # type: (Bool) -> str

        # do sanity check
        assert self.inited, "compiler not inited"
        assert self._ast_root is not None, "compiler.ast is None"
        assert isinstance(
            self._ast_root, ast.AST), "compiler.ast is not of type ast.AST"
        return ast.dump(self._ast_root)

    def dump(self, pretty=True, dump_file=""):
        # type: (Bool) -> str

        # do sanity check
        assert self.inited, "compiler not inited"
        assert self._ast_root is not None, "compiler.ast is None"
        assert isinstance(
            self._ast_root, ast.AST), "compiler.ast is not of type ast.AST"

        # dump with raw or formatted way
        return astunparse.dump(self._ast_root)

    def dump_src(self, pretty=True, dump_file=""):
        # type: (Bool) -> str

        # do sanity check
        assert self.inited, "compiler not inited"
        assert self._ast_root is not None, "compiler.ast is None"
        assert isinstance(
            self._ast_root, ast.AST), "compiler.ast is not of type ast.AST"

        # dump with raw or formatted way
        return self.src

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

        assert isinstance(
            ast_root, ast.AST), "compiler.ast is not of type ast.AST"
        self._ast_root = ast_root
        return

    def fix_missing_imports(self):
        print("dummy fix_missing_imports")
        _pass_manager = FixImportPassManager()
        _pass_manager.register_passes()
        _pass_manager.run(self)
        _cpass_list = _pass_manager.concrete_pass
        _src_list = None
        for _cpass in _cpass_list:
            _src_list = _cpass.import_vector
        for _src in _src_list:
            self._import_code_vector.append(_src)

        return
