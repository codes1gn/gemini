
import ast

from gemini.utils import *

from ..transformer.plugin_import_fix_transformer import PluginImportFixTransformer
from .pass_base import PassBase

__all__ = [
    'PluginImportFixPass',
]


class PluginImportFixPass(PassBase):

    __slots__ = [
        '_solvers',
    ]

    def __init__(self):
        # type: (None) -> None
        super(PluginImportFixPass, self).__init__()
        self._solvers = []
        self._solvers.append(PluginImportFixTransformer)

    def _build_plugin_module(self, _ast):
        _node = ast.Import(
            names=[
                ast.alias(
                    name='gemini.plugins.bert_plugin',
                    asname='gemini_plugin'
                )
            ]
        )
        _node2 = ast.ImportFrom(
            module='gemini.plugins.monad',
            names=[
                ast.alias(
                    name='MonadicTensor',
                    asname=None
                )
            ],
            level=0
        )
        # FIXME can we do it more flexible
        _ast.body.insert(5, _node)
        _ast.body.insert(6, _node2)

        ast.fix_missing_locations(_ast)
        return _ast


    def run_pass(self, _cnode):
        solver1 = self._solvers[0]()

        _cnode.ast = self._build_plugin_module(_cnode.ast)

        if _cnode._has_sub_nodes():
            for _sub_cnode in _cnode.sub_code_nodes:
                _sub_cnode.ast = self._build_plugin_module(_sub_cnode.ast)

        _cnode.dump(pretty=True, prefix='apply_import_fix_pass')

        return _cnode
