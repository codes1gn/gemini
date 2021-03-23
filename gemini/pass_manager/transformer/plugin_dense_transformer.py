
import ast

from gemini.utils import *
from .node_transformer_base import NodeTransformerBase

__all__ = [
    'PluginDenseTransformer',
]


class PluginDenseTransformer(NodeTransformerBase):

    __slots__ = [
    ]

    def __init__(self):
        super(self.__class__, self).__init__()

    def visit_Call(self, node):
        parent_node = node.gemini_parent

        if isinstance(node, ast.Call) and \
                hasattr(node.func, 'value') and \
                hasattr(node.func.value, 'value') and \
                hasattr(node.func.value.value, 'id') and \
                node.func.value.value.id == 'tf' and \
                hasattr(node.func.value, 'attr') and \
                node.func.value.attr == 'layers' and \
                hasattr(node.func, 'attr') and \
                node.func.attr == 'dense':
            # print 'found a tf.transpose, convert it to gemini_plugin.transpose'
            def _check_parent(_node):
                # check if my parent is pooler context
                # if parent is None, parent cannot be pooler context, return
                # False
                if not hasattr(_node, 'gemini_parent'):
                    return False

                _pnode = _node.gemini_parent
                if isinstance(_pnode, ast.With) and \
                        isinstance(_pnode.context_expr, ast.Call) and \
                        isinstance(_pnode.context_expr.args[0], ast.Str) and \
                        _pnode.context_expr.args[0].s == 'pooler':
                    # _pnode is pooler context
                    return True
                else:
                    _ret = _check_parent(_pnode)
                    return _ret

            _if_skip = _check_parent(parent_node)
            print _if_skip
            if _if_skip:
                print('skipping current dense node under pooler variable_scope')
                return node

            # if not pooler's dense, do norm sharding
            _i_node = ast.Name(
                id='gemini_plugin',
                ctx=ast.Load()
            )
            node.func.value = _i_node

        ast.fix_missing_locations(node)
        return node
