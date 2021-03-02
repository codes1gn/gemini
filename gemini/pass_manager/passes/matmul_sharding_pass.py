from gemini.pass_manager.transformer import *
from gemini.utils import *

from .pass_base import PassBase

__all__ = [
    'MatmulShardingPass',
]


class MatmulShardingPass(PassBase):

    __slots__ = [
        'solvers',
    ]

    def __init__(self):
        # type: (None) -> None
        super(MatmulShardingPass, self).__init__()
        self.solvers = []
        self.solvers.append(MatmulShardingOperationTransformer)
        self.solvers.append(MatmulShardingOperandTransformer)

    # method to run pass on ast tree
    def run_pass(self, _cnode):
        # run passes by sequential
        solver1 = self.solvers[0](sharding_size=2)
        solver2 = self.solvers[1](solver1)
        _cnode.ast = solver1.visit(_cnode.ast)
        solver2.visit(_cnode.ast)

        if _cnode._has_sub_nodes():
            for _sub_node in _cnode.sub_code_nodes:
                _sub_node.ast = solver1.visit(_sub_node.ast)
                solver2.visit(_sub_node.ast)

        return _cnode
