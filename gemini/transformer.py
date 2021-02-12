import ast

__all__ = ['Transformer']


class Transformer(ast.NodeTransformer):

    # hahaa
    def visit_Add(self, node):
        if node.id == 'Add':
            return ast.Sub()
        return node
