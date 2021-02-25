
import ast

from gemini.pass_manager.transformer import *
from .pass_base import PassBase

__all__ = [
    'ReadImportPass',
]


class ReadImportPass(PassBase):

    __slots__ = [
        '_solvers',
        '_import_vector',
    ]

    @property
    def import_vector(self):
        return self._import_vector

    def __init__(self):
        # type: (None) -> None
        super(ReadImportPass, self).__init__()
        self._solvers = []
        self._solvers.append(ReadImportTransformer)

    # method to run pass on ast tree
    def run_ast(self, _ast):
        # run passes by sequential
        solver1 = self._solvers[0]()
        # solver2 = self.solvers[1](solver1)
        # _ast = solver0.visit(_ast)
        solver1.visit(_ast)
        self._import_vector = solver1._import_vector

        assert isinstance(_ast, ast.AST)
        return _ast
