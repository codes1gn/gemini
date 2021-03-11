
import ast

from gemini.utils import *

from ..transformer.plugin_layer_norm_transformer import PluginLayerNormTransformer
from .pass_base import PassBase

__all__ = [
    'PluginLayerNormPass',
]


class PluginLayerNormPass(PassBase):

    __slots__ = [
        '_solvers',
    ]

    def __init__(self):
        # type: (None) -> None
        super(PluginLayerNormPass, self).__init__()
        self._solvers = []
        self._solvers.append(PluginLayerNormTransformer)

    def run_pass(self, _cnode):
        solver1 = self._solvers[0]()

        _cnode.ast = solver1.visit(_cnode.ast)
        ast.fix_missing_locations(_cnode.ast)

        if _cnode._has_sub_nodes():
            for _sub_cnode in _cnode.sub_code_nodes:
                _sub_cnode.ast = solver1.visit(_sub_cnode.ast)
                ast.fix_missing_locations(_sub_cnode.ast)

        _cnode.dump(pretty=True, prefix='apply_layer_norm_pass')
        return _cnode
