import os
import ast
import graphviz as gv
import numbers
import re
from uuid import uuid4 as uuid

__all__ = [
    'ast_to_dot'
]

# def transformed_ast(ast_root):


def ast_to_dot(ast_root, dot_label):
    assert(isinstance(ast_root, ast.AST))
    ast_transformed = transform_ast(ast_root)
    renderer = GraphRenderer()
    renderer.render(ast_transformed, label=dot_label)
    return ast_transformed


def to_camelcase(string):
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


def transform_ast(ast_node):
    if isinstance(ast_node, ast.AST):
        node = {to_camelcase(k): transform_ast(getattr(ast_node, k))
                for k in ast_node._fields}
        node['node_type'] = to_camelcase(ast_node.__class__.__name__)
        return node
    elif isinstance(ast_node, list):
        return [transform_ast(el) for el in ast_node]
    else:
        return ast_node


class GraphRenderer:
    """
    this class is capable of rendering data structures consisting of
    dicts and lists as a graph using graphviz
    """

    graphattrs = {
        'labelloc': 't',
        'fontcolor': 'white',
        'bgcolor': '#333333',
        'margin': '0',
    }

    nodeattrs = {
        'color': 'white',
        'fontcolor': 'white',
        'style': 'filled',
        'fillcolor': '#006699',
    }

    edgeattrs = {
        'color': 'white',
        'fontcolor': 'white',
    }

    _graph = None
    _rendered_nodes = None

    @staticmethod
    def _escape_dot_label(str):
        return str.replace("\\", "\\\\").replace(
            "|", "\\|").replace("<", "\\<").replace(">", "\\>")

    def _render_node(self, node):
        if isinstance(node, (str, numbers.Number)) or node is None:
            node_id = uuid()
        else:
            node_id = id(node)
        node_id = str(node_id)

        if node_id not in self._rendered_nodes:
            self._rendered_nodes.add(node_id)
            if isinstance(node, dict):
                self._render_dict(node, node_id)
            elif isinstance(node, list):
                self._render_list(node, node_id)
            else:
                self._graph.node(
                    node_id, label=self._escape_dot_label(
                        str(node)))

        return node_id

    def _render_dict(self, node, node_id):
        self._graph.node(node_id, label=node.get("node_type", "[dict]"))
        for key, value in node.items():
            if key == "node_type":
                continue
            child_node_id = self._render_node(value)
            self._graph.edge(
                node_id,
                child_node_id,
                label=self._escape_dot_label(key))

    def _render_list(self, node, node_id):
        self._graph.node(node_id, label="[list]")
        for idx, value in enumerate(node):
            child_node_id = self._render_node(value)
            self._graph.edge(
                node_id,
                child_node_id,
                label=self._escape_dot_label(
                    str(idx)))

    def render(self, data, label=None):
        # create the graph
        graphattrs = self.graphattrs.copy()
        if label is not None:
            graphattrs['label'] = self._escape_dot_label(label)
        graph = gv.Digraph(
            graph_attr=graphattrs,
            node_attr=self.nodeattrs,
            edge_attr=self.edgeattrs)

        # recursively draw all the nodes and edges
        self._graph = graph
        self._rendered_nodes = set()
        self._render_node(data)
        self._graph = None
        self._rendered_nodes = None

        # display the graph
        graph.format = "pdf"
        if bool(os.getenv("DEBUG_MODE", "False").lower() in ['true', '1']):
            graph.view(filename='ast.gv', directory='dumps/gv/')
        else:
            graph.render(filename='ast.gv', directory='dumps/gv/', view=False)
