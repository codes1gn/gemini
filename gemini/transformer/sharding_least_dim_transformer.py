import ast
import astunparse
from .base_transformer import *
from gemini.utils import *

__all__ = ['ShardingLeastDimTransformer']


class ShardingLeastDimTransformer(BaseTransformer):

    __slots__ = [
        '_sharding_size',
        '_split_weights',
    ]

    def __init__(self, sharding_size=1):
        self._sharding_size = sharding_size
        self._split_weights = {'left': [], 'right': []}
        super(BaseTransformer, self).__init__()

    @property
    def sharding_size(self):
        return self._sharding_size

    @property
    def split_weights(self):
        # type: (None) -> dict
        return self._split_weights

    def visit_BinOp(self, node):
        # print(astunparse.dump(node))
        # sanity check, do shortcut if not change
        if self._sharding_size == 1:
            return node

        # situation one, tf.matmul(a, b) + c
        if isinstance(node.left, ast.Call) and hasattr(
                node.left.func, 'attr') and (node.left.func.attr == "matmul"):
            # print('visiting ' + node.left.func.attr)
            lhs_id = node.left.args[0].id
            rhs_id = node.left.args[1].id

            # handle split weights, add weights id to the list
            self._split_weights['right'].append(rhs_id)
            self._split_weights['left'].append(lhs_id)
            # print('lhs id = ' + lhs_id)
            # print('rhs id = ' + rhs_id)
            func_attr = ast.Attribute(
                value=ast.Name(
                    id='tf',
                    ctx=ast.Load()
                ),
                attr='matmul',
                ctx=ast.Load()
            )
            reduce_attr = ast.Attribute(
                value=ast.Name(
                    id='tf',
                    ctx=ast.Load()
                ),
                attr='add_n',
                ctx=ast.Load()
            )
            _tmp = []
            for i in range(self._sharding_size):
                lhs_op = ast.Name(id=lhs_id + '_{}'.format(i), ctx=ast.Load())
                rhs_op = ast.Name(id=rhs_id + '_{}'.format(i), ctx=ast.Load())
                _tmp.append(ast.Call(
                    func=func_attr,
                    args=[
                        lhs_op,
                        rhs_op
                    ],
                    keywords=[],
                    starargs=None,
                    kwargs=None
                ))
            ret_node = ast.Call(
                func=reduce_attr,
                args=[ast.List(elts=_tmp, ctx=ast.Load())],
                keywords=[],
                starargs=None,
                kwargs=None
            )
            # ast_analysis(ret_node)
            node.left = ret_node

        ast.fix_missing_locations(node)
        return node
