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

from .code_tree.code_node_root import CodeNodeRoot

__all__ = [
    'GeminiCompiler',
]


class GeminiCompiler:

    __slots__ = [
        '_pass_manager',
        '_import_code_vector',
        '_code_node_entry',
        '_env',
    ]

    def __init__(self):
        self._pass_manager = None
        self._model_code_path = None
        self._import_code_vector = []
        self._code_node_entry = CodeNodeRoot()
        self._env = globals

    @property
    def ast(self):
        return self._code_node_entry.ast

    @property
    def import_code_vector(self):
        return self._import_code_vector

    @property
    def src(self):
        try:
            self._code_node_entry.src = astunparse.unparse(
                self._code_node_entry.ast)
        except Exception:
            assert 0, 'unparse ast_root failed, cannot update source_code'
        return self._code_node_entry.src

    @property
    def inited(self):
        return True if self.ast is not None else False

    @property
    def src_file(self):
        return self._code_node_entry.src_file

    # method to apply MP patterns
    def apply_model_parallel(self, config):
        # TODO(albert) add config class
        if config['mode'] == "sharding":
            self._pass_manager = ShardingPassManager()
            self._pass_manager.register_passes()
            self._pass_manager.run(self._code_node_entry)

    def compile_and_run(self, use_ast=False):
        # print('global keys have\n')
        # print(globals().keys())
        # TODO defaultly, use src to run, rather than ast
        assert self.inited, "compiler not inited"
        # assert use_ast == False, "exec with ast is NotImplemented yet"
        if use_ast == False:
            if self._code_node_entry._has_sub_nodes():
                for _sub_node in self._code_node_entry.sub_code_nodes:
                    code_obj = compile(
                        _sub_node.src,
                        filename=_sub_node.get_module_name(),
                        mode='exec')
                    _module = types.ModuleType("import_lib", "import_lib doc")
                    exec(code_obj, _module.__dict__)
                    self._env()['import_lib'] = _module

            exec(self._code_node_entry.src, self._env())
        # TODO fix ast run bugs.
        # elif use_ast == True:
        #     assert isinstance(
        #         self._code_node_entry.ast, ast.AST), "expected ast.AST, but got " + str(type(self._code_node_entry.ast))
        #     co_obj = compile(
        #         self._code_node_entry.ast,
        #         filename=self._code_node_entry.src_file,
        #         mode='exec')
        #     exec(co_obj, environment)

        pass

    def dump(self, pretty=True, prefix="anonymous"):
        self._code_node_entry.dump(pretty=pretty, prefix=prefix)

    def dump_src(self, prefix="anonymous"):
        self._code_node_entry.dump_src(prefix=prefix)

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
            self._code_node_entry.src_file = src_filename
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
            self._code_node_entry.src_file = src_filename
            src_code = func_or_src
            ast_root = ast.parse(src_code, filename=src_filename)

        assert isinstance(
            ast_root, ast.AST), "compiler.ast is not of type ast.AST"
        self._code_node_entry.ast = ast_root
        return

    def parse_modules(self):
        # STEP 1: read import source codes
        _pass_manager = ImportModulePassManager()
        _pass_manager.register_passes()
        _pass_manager.run(self._code_node_entry)
        return
