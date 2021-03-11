
import ast

from gemini.utils import *

from ..transformer.plugin_dense_transformer import PluginDenseTransformer
from .pass_base import PassBase

__all__ = [
    'PluginDensePass',
]


class PluginDensePass(PassBase):

    __slots__ = [
        '_solvers',
    ]

    def __init__(self):
        # type: (None) -> None
        super(PluginDensePass, self).__init__()
        self._solvers = []
        self._solvers.append(PluginDenseTransformer)

    def run_pass(self, _cnode):
        solver1 = self._solvers[0]()

        _cnode.ast = solver1.visit(_cnode.ast)
        ast.fix_missing_locations(_cnode.ast)

        if _cnode._has_sub_nodes():
            for _sub_cnode in _cnode.sub_code_nodes:
                _sub_cnode.ast = solver1.visit(_sub_cnode.ast)
                ast.fix_missing_locations(_sub_cnode.ast)

        _cnode.dump(pretty=True, prefix='apply_dense_pass')
        return _cnode
