import inspect
import os
import textwrap
import ast
import astunparse
import importlib
import types

from typing import Callable

from gemini.pass_manager import *
from gemini.utils import *
from gemini.code_tree import *

__all__ = [
    'GeminiCompiler',
]


class GeminiCompiler:

    __slots__ = [
        # '_code_node_entry.ast',
        # '_code_node_entry.src',
        '_src_file',
        '_pass_manager',
        '_import_code_vector',
        '_code_node_entry',
    ]

    def __init__(self):
        # self._code_node_entry.ast = None
        # self._code_node_entry.src = ""
        self._src_file = ""
        self._pass_manager = None
        self._import_code_vector = []
        self._code_node_entry = CodeNodeRoot()

    @property
    def ast(self):
        return self._code_node_entry.ast

    @property
    def import_code_vector(self):
        return self._import_code_vector

    @property
    def src(self):
        try:
            self._code_node_entry.src = astunparse.unparse(self._code_node_entry.ast)
        except Exception:
            assert 0, 'unparse ast_root failed, cannot update source_code'
        return self._code_node_entry.src

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
                self._code_node_entry.ast, ast.AST), "expected ast.AST, but got " + str(type(self._code_node_entry.ast))
            co_obj = compile(
                self._code_node_entry.ast,
                filename=self._src_file,
                mode='exec')
            exec(co_obj, environment)
        else:
            # way to dynamically load module like import
            # mod = importlib.import_module('import_lib')
            code_obj = compile(self._import_code_vector[0], filename='import_lib', mode='exec')
            _module = types.ModuleType("import_lib", "import_lib doc")
            exec(code_obj, _module.__dict__)
            # assert 0
            environment['import_lib'] = _module

            exec(self._code_node_entry.src, environment)
        pass

    def raw_dump(self):
        # type: (Bool) -> str

        # do sanity check
        assert self.inited, "compiler not inited"
        assert self._code_node_entry.ast is not None, "compiler.ast is None"
        assert isinstance(
            self._code_node_entry.ast, ast.AST), "compiler.ast is not of type ast.AST"
        return ast.dump(self._code_node_entry.ast)

    def dump(self, pretty=True, dump_file=""):
        # type: (Bool) -> str

        # do sanity check
        assert self.inited, "compiler not inited"
        assert self._code_node_entry.ast is not None, "compiler.ast is None"
        assert isinstance(
            self._code_node_entry.ast, ast.AST), "compiler.ast is not of type ast.AST"

        # dump with raw or formatted way
        return astunparse.dump(self._code_node_entry.ast)

    def dump_src(self, pretty=True, dump_file=""):
        # type: (Bool) -> str

        # do sanity check
        assert self.inited, "compiler not inited"
        assert self._code_node_entry.ast is not None, "compiler.ast is None"
        assert isinstance(
            self._code_node_entry.ast, ast.AST), "compiler.ast is not of type ast.AST"

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
        self._code_node_entry.ast = ast_root
        return

    def fix_missing_imports(self):
        # STEP 1: read import source codes
        print("dummy fix_missing_imports")
        _pass_manager = ReadImportPassManager()
        _pass_manager.register_passes()
        _pass_manager.run(self)
        _cpass_list = _pass_manager.concrete_pass
        _src_list = None
        for _cpass in _cpass_list:
            _src_list = _cpass.import_vector
        for _src in _src_list:
            self._import_code_vector.append(_src)

        # STEP 2: transform a bit
        _pass_manager = None
        _pass_manager = ModuleTransPassManager()
        _pass_manager.register_passes()
        _pass_manager.run(self)

        # STEP 3: load import modules
        return
