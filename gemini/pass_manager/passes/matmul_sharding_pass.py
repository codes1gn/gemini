
import ast

from gemini.pass_manager.transformer import *
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
    def run_ast(self, _ast):
        # run passes by sequential
        # solver0 = self.solvers[0]()
        solver1 = self.solvers[0](sharding_size=2)
        solver2 = self.solvers[1](solver1)
        # _ast = solver0.visit(_ast)
        _ast = solver1.visit(_ast)
        solver2.visit(_ast)

        assert isinstance(_ast, ast.AST)
        return _ast
