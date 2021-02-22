import ast

from gemini.utils import *
from .base_transformer import BaseTransformer

__all__ = ['SetParentTransformer']


class SetParentTransformer(BaseTransformer):

    def generic_visit(self, node):
        """
        printing visit messages
        """
        super(BaseTransformer, self).generic_visit(node)
        for child in ast.iter_child_nodes(node):
            setattr(child, 'gemini_parent', node)
            assert hasattr(child, 'gemini_parent')
        return node
