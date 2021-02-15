import ast

__all__ = ['BaseTransformer']


class BaseTransformer(ast.NodeTransformer):

    def generic_visit(self, node):
        """
        printing visit messages
        """
        super(ast.NodeTransformer, self).generic_visit(node)
        print("anchor BaseTransformer do generic visit -- " + node.__class__.__name__)
        return node

    def visit_Assign(self, node):
        print('anchor BaseTransformer visit_Assign', ast.dump(node))
        # if node.id == 'Add':
        #     return ast.Sub()
        return node

    def visit_Name(self, node):
        print('anchor BaseTransformer visit_Name', ast.dump(node))
        return node

    def visit_Call(self, node):
        print('anchor BaseTransformer visit_Call', ast.dump(node))
        return node

