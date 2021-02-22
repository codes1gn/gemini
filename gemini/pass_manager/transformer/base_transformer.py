import ast
from gemini.utils import *

__all__ = ['BaseTransformer']


class BaseTransformer(ast.NodeTransformer):

    def generic_visit(self, node):
        """
        printing visit messages
        """
        super(ast.NodeTransformer, self).generic_visit(node)
        vlog(
            "BaseTransformer generic_visit -- " +
            node.__class__.__name__)
        return node
