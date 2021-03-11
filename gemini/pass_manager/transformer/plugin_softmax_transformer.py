
import ast

from gemini.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    'PluginSoftmaxTransformer',
]


class PluginSoftmaxTransformer(NodeTransformerBase):

    __slots__ = [
    ]

    def __init__(self):
        super(self.__class__, self).__init__()

    def visit_Assign(self, node):
        parent_node = node.gemini_parent

        if isinstance(node.value, ast.Call) and hasattr(
                node.value.func, 'attr') and node.value.func.attr == 'matmul':
            pretty_dump(parent_node)

        ast.fix_missing_locations(node)
