
import ast

from gemini.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    'ModuleTransDemoAlphaTransformer',
]


class ModuleTransDemoAlphaTransformer(NodeTransformerBase):

    __slots__ = [
    ]

    def __init__(self, sharding_size=1):
        super(NodeTransformerBase, self).__init__()

    def visit_Print(self, node):
        # print(astunparse.dump(node))
        # sanity check, do shortcut if not change
        node.values[0].s='world'

        ast.fix_missing_locations(node)
        return node
