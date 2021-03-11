
import ast

from gemini.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    'PluginImportFixTransformer',
]


class PluginImportFixTransformer(NodeTransformerBase):

    __slots__ = [
    ]

    def __init__(self):
        super(self.__class__, self).__init__()

    def visit_Module(self, node):
        # import_module_node = ast.Import(
        #     names=ast.alias(
        #         name=[
        #             name='gemini.plugins.bert_plugin',
        #             asname='gemini_plugin'
        #         ]
        #     )
        # )

        # pretty_dump(node)
        # assert 0, 'debug'
        # ast.fix_missing_locations(node)
        return node


