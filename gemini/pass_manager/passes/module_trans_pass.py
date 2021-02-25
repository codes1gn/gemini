import astunparse
import ast

from gemini.pass_manager.transformer import *
from .pass_base import PassBase
from gemini.utils import *

__all__ = [
    'ModuleTransPass',
]


class ModuleTransPass(PassBase):

    __slots__ = [
        'solvers',
    ]

    def __init__(self):
        # type: (None) -> None
        super(ModuleTransPass, self).__init__()
        self.solvers = []
        self.solvers.append(ModuleTransDemoAlphaTransformer)
        # self.solvers.append(ModuleTransOperandTransformer)

    # method to run pass on ast tree
    def run_ast(self, _ast):
        # run passes by sequential
        solver0 = self.solvers[0]()
        _ast = solver0.visit(_ast)

        assert isinstance(_ast, ast.AST)
        return _ast

    # method to run pass on ast tree
    def run_src(self, _src):
        # run passes by sequential
        solver0 = self.solvers[0]()
        _ast = ast.parse(_src)
        _ast = solver0.visit(_ast)

        assert isinstance(_ast, ast.AST)
        new_src = astunparse.unparse(_ast)
        return new_src
