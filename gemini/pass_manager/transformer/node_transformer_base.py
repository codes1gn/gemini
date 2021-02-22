import ast

from gemini.utils import *


class NodeTransformerBase(ast.NodeTransformer):

    def generic_visit(self, node):
        """
        printing visit messages
        """
        self.fix_missing_parent(node)
        super(ast.NodeTransformer, self).generic_visit(node)
        vlog(
            "BaseTransformer generic_visit -- " +
            node.__class__.__name__)
        # for child in ast.iter_child_nodes(node):
        #     setattr(child, 'gemini_parent', node)
        #     assert hasattr(child, 'gemini_parent')
        return node

    def fix_missing_parent(self, node):
        for child in ast.iter_child_nodes(node):
            setattr(child, 'gemini_parent', node)
            assert hasattr(child, 'gemini_parent')
