import ast
from .base_transformer import *

__all__ = ['ShardingLeastDimTransformer']


class ShardingLeastDimTransformer(BaseTransformer):

    def visit_Assign(self, node):
        print('anchor ShardingLeastDimTransformer visit_Assign', ast.dump(node))
        # if node.id == 'Add':
        #     return ast.Sub()
        return node

    def visit_Name(self, node):
        print('anchor ShardingLeastDimTransformer visit_Name', ast.dump(node))
        return node

    def visit_Call(self, node):
        print('anchor ShardingLeastDimTransformer visit_Call', ast.dump(node))
        return node

