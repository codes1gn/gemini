
import ast

from gemini.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    'PluginMultiplyTransformer',
]


class PluginMultiplyTransformer(NodeTransformerBase):

    __slots__ = [
    ]

    def __init__(self):
        super(self.__class__, self).__init__()

    def visit_Call(self, node):
        parent_node = node.gemini_parent

        if isinstance(node, ast.Call) and \
                hasattr(node.func, 'value') and \
                hasattr(node.func.value, 'id') and \
                node.func.value.id == 'tf' and \
                hasattr(node.func, 'attr') and \
                node.func.attr == 'multiply':
            # print 'found a tf.transpose, convert it to gemini_plugin.transpose'
            node.func.value.id = 'gemini_plugin'

        ast.fix_missing_locations(node)
        return node
