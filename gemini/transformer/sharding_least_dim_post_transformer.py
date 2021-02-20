import ast
import astunparse
from .base_transformer import *
from . import *
from gemini.utils import *

__all__ = ['ShardingLeastDimPostTransformer']


class ShardingLeastDimPostTransformer(ast.NodeVisitor):

    __slots__ = [
        '_sharding_size',
        '_split_weights',
    ]

    def __init__(self, tfr):
        assert isinstance(tfr, ShardingLeastDimTransformer),\
            "call __init__ of ShardingLeastDimPostTransformer; transformer is not of type ShardingLeastDimTransformer"
        self._sharding_size = tfr.sharding_size
        self._split_weights = tfr.split_weights
        super(ast.NodeVisitor, self).__init__()

    @property
    def sharding_size(self):
        return self._sharding_size

    @property
    def split_weights(self):
        return self._split_weights

    def visit_Assign(self, node):
        # vlog(astunparse.dump(node))
        # vlog(astunparse.dump(node.targets[0]))
        assert hasattr(node, 'gemini_parent'), "split_weights not have parents"
        parent_node = node.gemini_parent
        if hasattr(
                node.targets[0], 'id') and node.targets[0].id in self._split_weights:
            print('before split weight')
            print(astunparse.dump(parent_node))
            print('-----------------------\n')

            # modify shape before copy
            print('before modify')
            print(astunparse.dump(node))
            print('-----------------------\n')
            # TODO check if weights are 2 dims, only handles matmul 2d
            assert len(
                node.value.args[0].elts) == 2, "yet, support matmul 2d only"
            node.value.args[0].elts[0] = ast.BinOp(
                left=node.value.args[0].elts[0],
                # TODO check if reduce dim is times of sharding size
                op=ast.FloorDiv(),
                right=ast.Num(n=self._sharding_size)
            )
            print('after modify')
            print(astunparse.dump(node))
            print('-----------------------\n')

            # add nodes copys to parent node
            import copy
            node_src = node
            node_dst = copy.deepcopy(node_src)

            # change id sequentially
            node_src.targets[0].id += '_0'
            node_dst.targets[0].id += '_1'

            print('clone src')
            print(astunparse.dump(node))
            print('-----------------------\n')

            print('clone dst')
            print(astunparse.dump(node_dst))
            print('-----------------------\n')

            old_index = parent_node.body.index(node)
            parent_node.body.insert(old_index + 1, node_dst)

            # set gemini_parent
            setattr(node_dst, 'gemini_parent', parent_node)

            # dump parent node after all done
            print('after split weight')
            print(astunparse.dump(parent_node))
            print('-----------------------\n')
            ast.fix_missing_locations(node)

        self.generic_visit(node)
