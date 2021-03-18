
import ast

from gemini.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    'BertDevicePlacerTransformer',
]


class BertDevicePlacerTransformer(NodeTransformerBase):

    __slots__ = [
        'config',
    ]

    def __init__(self):
        self.config = Configuration()
        super(self.__class__, self).__init__()

    def visit_Call(self, node):
        parent_node = node.gemini_parent
        if isinstance(parent_node, ast.With) and \
                node.func.attr == 'variable_scope':
            print('hahaha')
            pretty_dump(node)
            # if isinstance(node.args[0], ast.Str) and 'embedding' in node.args[0].s:
            #     _key_name = node.args[0].s
            #     print('ast.With', node.func.attr, ' ', _key_name)
            #     _device_str = self.config.get_device_by_tensor_name(_key_name)
            #     _dev_node = ast.With(
            #         context_expr = ast.Call(
            #             func = ast.Attribute(
            #                 value = ast.Name(
            #                     id = 'tf',
            #                     ctx = ast.Load()
            #                 ),
            #                 attr='device',
            #                 ctx=ast.Load()
            #             ),
            #             args = [
            #                 ast.Str(
            #                     s = _device_str
            #                 )
            #             ],
            #             keywords=[],
            #             starargs=None,
            #             kwargs=None
            #         ),
            #         optional_vars=None,
            #         body=parent_node.body
            #     )
            #     parent_node.body=[_dev_node]
            #     pretty_dump(parent_node)

        ast.fix_missing_locations(node)
        return node
