
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
            pretty_dump(node)

            if isinstance(node.args[0], ast.Str) and 'embedding' in node.args[0].s:

                _key_name = node.args[0].s
                _device_str = self.config.get_device_by_tensor_name(_key_name)
                _dev_node = ast.With(
                    context_expr = ast.Call(
                        func = ast.Attribute(
                            value = ast.Name(
                                id = 'tf',
                                ctx = ast.Load()
                            ),
                            attr='device',
                            ctx=ast.Load()
                        ),
                        args = [
                            ast.Str(
                                s = _device_str
                            )
                        ],
                        keywords=[],
                        starargs=None,
                        kwargs=None
                    ),
                    optional_vars=None,
                    body=parent_node.body
                )
                parent_node.body=[_dev_node]

            elif isinstance(node.args[0], ast.BinOp) and \
                isinstance(node.args[0].left, ast.Str) and \
                node.args[0].left.s == "layer_%d":

                assert isinstance(parent_node, ast.With), 'expect parent is with tf.device'
                old_body = parent_node.body
                # import astunparse
                # print(astunparse.unparse(old_body))
                # assert 0, 'debug'
                new_body = ast.With(
                    context_expr = ast.Call(
                        func = ast.Attribute(
                            value = ast.Name(
                                id = 'tf',
                                ctx = ast.Load()
                            ),
                            attr='device',
                            ctx=ast.Load()
                        ),
                        args = [
                            ast.Call(
                                func = ast.Attribute(
                                    value = ast.Name(
                                        id = 'gemini_config',
                                        ctx = ast.Load()
                                    ),
                                    attr='get_device_by_tensor_name',
                                    ctx=ast.Load()
                                ),
                                args = [
                                    ast.BinOp(
                                        left=ast.Str(s='layer_%d'),
                                        op=ast.Mod(),
                                        right=ast.Name(
                                            id='layer_idx',
                                            ctx=ast.Load()
                                        )
                                    )
                                ],
                                keywords=[],
                                starargs=None,
                                kwargs=None
                            )
                        ],
                        keywords=[],
                        starargs=None,
                        kwargs=None
                    ),
                    optional_vars=None,
                    body=old_body
                )
                parent_node.body = new_body

        ast.fix_missing_locations(node)
        return node
