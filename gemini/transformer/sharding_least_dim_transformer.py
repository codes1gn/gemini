import ast
import astunparse
from .base_transformer import *

__all__ = ['ShardingLeastDimTransformer']


class ShardingLeastDimTransformer(BaseTransformer):

    # def visit_Call(self, node):
    #     print(astunparse.dump(node))
    #     return node
    __slots__ = [
        '_sharding_size',
    ]

    def __init__(self, sharding_size=1):
        self._sharding_size = sharding_size
        super(BaseTransformer, self).__init__()


    def visit_BinOp(self, node):
        # print(astunparse.dump(node))

        # situation one, tf.matmul(a, b) + c
        if isinstance(node.left, ast.Call) and hasattr(node.left.func, 'attr') and (node.left.func.attr == "matmul"):
            print('visiting ' + node.left.func.attr)
            lhs_id = node.left.args[0].id
            rhs_id = node.left.args[1].id
            print('lhs id = ' + lhs_id)
            print('rhs id = ' + rhs_id)
            func_attr=ast.Attribute(
                value=ast.Name(
                    id='tf',
                    ctx=ast.Load()
                ),
                attr='matmul',
                ctx=ast.Load()
            )
            lhs_op = ast.Name(id=lhs_id+'new', ctx=ast.Load())
            rhs_op = ast.Name(id=rhs_id+'new', ctx=ast.Load())
            ret_node = ast.Call(
                func=func_attr,
                args=[
                    lhs_op,
                    rhs_op
                ]
            )
            print(astunparse.dump(node))
            node.left = ret_node
            print(astunparse.dump(node))

        return node

