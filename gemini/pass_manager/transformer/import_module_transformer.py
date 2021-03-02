import ast
import copy
import sys
import importlib
import inspect

from tops_models.common_utils import get_python_version

from gemini.utils import *

from .node_transformer_base import NodeTransformerBase

__all__ = [
    'ImportModuleTransformer',
]


# TODO(albert) handle alias
# TODO(albert) fix alias source code insertion
# TODO(albert) enable ImportFrom
# TODO(albert) fix ImportFrom *
# TODO(albert) blacklist not implemented, internal packages need to be handled
class ImportModuleTransformer(ast.NodeTransformer):

    __slots__ = [
        '_modules',
        '_blacklist',
        '_python_builtin_packages',
    ]

    def __init__(self):
        self._modules = {}
        self._blacklist = [
            # TODO(albert), ban module by top level name
            'horovod',
            'horovod.tensorflow',
            'tensorflow',
            'tops_models.tf_utils',
            'tops_models.common_utils',
            'tops_models.estimator_utils',
            'tops_models.logger',
        ]
        _syspath_bak = copy.deepcopy(sys.path)
        for _path in _syspath_bak:
            if 'python' not in _path and \
                '/usr/local' not in _path:
                sys.path.remove(_path)
        self._python_builtin_packages = get_python_library()
        sys.path = _syspath_bak
        del _syspath_bak

        super(ImportModuleTransformer, self).__init__()

    @property
    def modules(self):
        return self._modules

    def visit_Import(self, node):
        # note that import or importfrom node does not have parent
        # TODO(albert) support recursive importing
        _ret_node = node
        for name_head in node.names:
            module_name = name_head.name
            module_alias = name_head.asname

            # skip import process if current module is system module
            # if module_name in get_python_library():
            if module_name in self._python_builtin_packages:
                print '------ skip import ', module_name
                continue
            # skip import process if in blacklist
            if module_name in self._blacklist:
                print '------ skip import ', module_name
                continue

            _module = importlib.import_module(module_name)
            source_code = inspect.getsource(_module)
            del _module
            print '------ handle import ', module_name
            pretty_dump(node)
            if module_alias is not None:
                self._modules[module_alias] = source_code
            else:
                self._modules[module_name] = source_code
            _ret_node = None

        return _ret_node

    # def visit_ImportFrom(self, node):
    #     # vlog(astunparse.dump(node))
    #     # vlog(astunparse.dump(node.targets[0]))
    #     # note that import or importfrom node does not have parent

    #     # print(astunparse.dump(node))
    #     # for name_head in node.names:
    #     #     module_name = name_head.name
    #     #     new_module = importlib.import_module(module_name)
    #     #     source_code = inspect.getsource(new_module)
    #     #     print source_code
    #     #     self._modules.append(source_code)

    #     assert 0, 'from ** import **, not implemented, please use Import instead'
    #     self.generic_visit(node)
