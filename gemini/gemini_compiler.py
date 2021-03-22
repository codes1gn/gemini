import inspect
import textwrap
import ast
import astunparse
import copy
import sys
from imp import new_module
import traceback
from typing import Callable

from gemini.pass_manager import *
from gemini.utils import *

from .code_tree.code_node_root import CodeNodeRoot

__all__ = [
    'GeminiCompiler',
]


class GeminiCompiler:
    """Compiler Class that compiles the target source code into python AST.
    Then apply model parallel passes onto the AST. The well-transformed AST will
    then be executed on the python interpreter with supervision.

    Returns:
        GeminiCompiler: returns the instance of this class
    """

    __slots__ = [
        '_pass_manager',
        '_import_code_vector',
        '_code_node_entry',
        '_global_config',
        '_env',
    ]

    def __init__(self):
        self._pass_manager = None
        self._model_code_path = None
        self._import_code_vector = []
        self._code_node_entry = CodeNodeRoot()
        self._global_config = None
        self._env = globals

    @property
    def import_code_vector(self):
        return self._import_code_vector

    @property
    def inited(self):
        return True if self.ast is not None else False

    @property
    def src_file(self):
        return self._code_node_entry.src_file

    # method to apply MP patterns
    def apply_model_parallel(self, config):
        # TODO(albert) add config class
        self._global_config = config
        if self._global_config.mode == Mode.SHARDING:
            self._pass_manager = ShardingPassManager()
            self._pass_manager.register_passes()
            self._pass_manager.run(self._code_node_entry)

        self._pass_manager = PipelinePassManager()
        self._pass_manager.register_passes()
        self._pass_manager.run(self._code_node_entry)
        # add pipeline mode as default

    def compile_and_run(self, use_ast=False):
        # print('global keys have\n')
        # print(globals().keys())
        # TODO(albert) fix ast run bugs
        # import tensorflow as tf
        # def del_all_flags(FLAGS):
        #     flags_dict = FLAGS._flags()
        #     keys_list = [keys for keys in flags_dict]
        #     for keys in keys_list:
        #         FLAGS.__delattr__(keys)
        # del_all_flags(tf.flags.FLAGS)
        # print tf.flags.FLAGS
        # assert use_ast == False, "exec with ast is NotImplemented yet"
        if use_ast == False:
            _main_module = new_module('__main__')
            sys.modules['__main__'] = _main_module
            _entry_backup = self._env()['__name__']
            self._env()['__name__'] = '__main__'
            self._env()['gemini_config'] = self._global_config
            _sys_argv_backup = copy.deepcopy(sys.argv)

            import inspect
            if self._code_node_entry._has_sub_nodes():
                for _sub_node in self._code_node_entry.sub_code_nodes:
                    _module_name = _sub_node.get_module_name()
                    code_obj = compile(
                        _sub_node.src,
                        filename=_module_name,
                        mode='exec')
                    # legacy codes, use types not imp
                    # _module = types.ModuleType(
                    #     _module_name, _module_name + " doc")

                    # FIXME handle plugin import
                    import types
                    # _bert_plugin = types.ModuleType('gemini.plugins.bert_plugin', 'dummy')
                    # _bert_plugin = new_module('gemini.plugins.bert_plugin')
                    # sys.modules['gemini_entry'] = _bert_plugin
                    # FIXME, no attr reshape
                    # _module.__dict__['gemini_entry'] = _bert_plugin

                    # _module = new_module(_module_name)
                    _module = types.ModuleType(
                        _module_name, _module_name + " doc")
                    try:
                        _module.__dict__['gemini_config'] = self._global_config
                        exec(code_obj, _module.__dict__)
                    except Exception as e:
                        traceback.print_exc()
                    # assert 0, inspect.getmembers(_module)

                    # self._env()[_module_name] = _module
                    sys.modules[_module_name] = _module
                    _main_module.__dict__[_module_name] = _module

            sys.argv = sys.argv[1:]
            main_code_obj = compile(
                self._code_node_entry.src,
                filename=self._code_node_entry.src_file,
                mode='exec'
            )
            _main_module.__dict__['gemini_config'] = self._global_config
            # print(_main_module.__dict__)
            exec(main_code_obj, _main_module.__dict__)
            sys.argv = copy.deepcopy(_sys_argv_backup)
            self._env()['__name__'] = _entry_backup
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
