import ast
import astunparse
from .base_transformer import *
from . import *
from gemini.utils import *

__all__ = ['ShardingLeastDimPostTransformer']


class ShardingLeastDimPostTransformer(BaseTransformer):

    # def visit_Call(self, node):
    #     print(astunparse.dump(node))
    #     return node
    __slots__ = [
        '_sharding_size',
        '_split_weights',
    ]

    def __init__(self, tfr):
        assert isinstance(tfr, ShardingLeastDimTransformer),\
            "call __init__ of ShardingLeastDimPostTransformer; transformer is not of type ShardingLeastDimTransformer"
        self._sharding_size = tfr.sharding_size
        self._split_weights = tfr.split_weights
        super(BaseTransformer, self).__init__()

    @property
    def sharding_size(self):
        return self._sharding_size

    @property
    def split_weights(self):
        return self._split_weights

    def visit_Assign(self, node):
        vlog('visiting Assign Node')
        vlog(astunparse.dump(node))
        vlog(astunparse.dump(node.targets[0]))
        if hasattr(node.targets[0], 'id') and node.targets[0].id in self._split_weights:
            shape_list = node.value.args[0]
            shape_elements = shape_list.elts
            # TODO check if weights are 2 dims, only handles matmul 2d
            assert len(shape_elements) == 2, "yet, support matmul 2d only"
            shape_node_to_split = node.value.args[0].elts[0]
            (astunparse.dump(shape_node_to_split))
            shape_node_to_split = ast.BinOp(
                left=shape_node_to_split,
                # TODO check if reduce dim is times of sharding size
                op=ast.FloorDiv(),
                right=ast.Num(n=self._sharding_size)
            )
            print(astunparse.dump(shape_node_to_split))
        return node
