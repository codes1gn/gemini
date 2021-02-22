import ast


class NodeVisitorBase(ast.NodeVisitor):

    def fix_missing_parent(self, node):
        for child in ast.iter_child_nodes(node):
            setattr(child, 'gemini_parent', node)
            assert hasattr(child, 'gemini_parent')
