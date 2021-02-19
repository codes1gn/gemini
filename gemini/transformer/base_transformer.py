import ast
import astunparse
from gemini.utils import *

__all__ = ['BaseTransformer']


class BaseTransformer(ast.NodeTransformer):

    def generic_visit(self, node):
        """
        printing visit messages
        """
        super(ast.NodeTransformer, self).generic_visit(node)
        vlog(
            "BaseTransformer do generic visit -- " +
            node.__class__.__name__)
        # print(astunparse.dump(node))
        return node

    # def visit_Assign(self, node):
    #     vlog('BaseTransformer visit_Assign', ast.dump(node))
    #     # if node.id == 'Add':
    #     #     return ast.Sub()
    #     return node

    # def visit_Name(self, node):
    #     vlog('BaseTransformer visit_Name', ast.dump(node))
    #     return node

    # def visit_Call(self, node):
    #     vlog('BaseTransformer visit_Call', ast.dump(node))
    #     return node

