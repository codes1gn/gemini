import astunparse
import ast
import copy
import importlib as impl
import inspect

from gemini.utils import *

from .node_transformer_base import NodeTransformerBase

__all__ = [
    'ReadImportTransformer',
]


# TODO(albert) handle alias
# TODO(albert) fix alias source code insertion
# TODO(albert) enable ImportFrom
# TODO(albert) fix ImportFrom *
# TODO(albert) blacklist not implemented, internal packages need to be handled
class ReadImportTransformer(ast.NodeTransformer):

    __slots__ = [
        '_import_vector',
        '_blacklist',
    ]

    def __init__(self):
        self._import_vector = []
        super(ReadImportTransformer, self).__init__()

    @property
    def import_vector(self):
        return self._import_vector

    def visit_Import(self, node):
        # vlog(astunparse.dump(node))
        # vlog(astunparse.dump(node.targets[0]))
        # note that import or importfrom node does not have parent
        for name_head in node.names:
            module_name = name_head.name
            new_module = impl.import_module(module_name)
            source_code = inspect.getsource(new_module)
            self._import_vector.append(source_code)

        return None

    # def visit_ImportFrom(self, node):
    #     # vlog(astunparse.dump(node))
    #     # vlog(astunparse.dump(node.targets[0]))
    #     # note that import or importfrom node does not have parent

    #     # print(astunparse.dump(node))
    #     # for name_head in node.names:
    #     #     module_name = name_head.name
    #     #     new_module = impl.import_module(module_name)
    #     #     source_code = inspect.getsource(new_module)
    #     #     print source_code
    #     #     self._import_vector.append(source_code)

    #     assert 0, 'from ** import **, not implemented, please use Import instead'
    #     self.generic_visit(node)

