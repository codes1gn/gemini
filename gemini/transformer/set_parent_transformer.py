import ast
import astunparse
from gemini.utils import *
from .base_transformer import BaseTransformer

__all__ = ['SetParentTransformer']


class SetParentTransformer(BaseTransformer):

    def generic_visit(self, node):
        """
        printing visit messages
        """
        print('im here, visit ' + str(node))
        super(BaseTransformer, self).generic_visit(node)
        for child in ast.iter_child_nodes(node):
            print('visit child ' + str(child) + ' of father ' + str(node))
            setattr(child, 'gemini_parent', node)
            assert hasattr(child, 'gemini_parent')
            if hasattr(child, 'gemini_parent'):
                print(str(child) + 's` father is ' + str(child.gemini_parent))
            else:
                print('not found parent of ' + str(child))

        return node

