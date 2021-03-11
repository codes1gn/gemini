
import ast

from gemini.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    'PluginLayerNormTransformer',
]


class PluginLayerNormTransformer(NodeTransformerBase):

    __slots__ = [
    ]

    def __init__(self):
        super(self.__class__, self).__init__()

    def visit_Call(self, node):
        parent_node = node.gemini_parent

        if isinstance(node, ast.Call) and \
                hasattr(node.func, 'value') and \
                hasattr(node.func.value, 'value') and \
                hasattr(node.func.value.value, 'value') and \
                hasattr(node.func.value.value.value, 'id') and \
                node.func.value.value.value.id == 'tf' and \
                hasattr(node.func.value.value, 'attr') and \
                node.func.value.value.attr == 'contrib' and \
                hasattr(node.func.value, 'attr') and \
                node.func.value.attr == 'layers' and \
                hasattr(node.func, 'attr') and \
                node.func.attr == 'layer_norm':
            # print 'found a tf.transpose, convert it to gemini_plugin.transpose'
            _i_node = ast.Name(
                id='gemini_plugin',
                ctx=ast.Load()
            )
            node.func.value = _i_node

        ast.fix_missing_locations(node)
        return node
