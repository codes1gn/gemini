
import ast

from gemini.utils import *
from gemini.code_tree.code_node_leaf import CodeNodeLeaf

from ..transformer.import_module_transformer import ImportModuleTransformer
from .pass_base import PassBase

__all__ = [
    'ImportModulePass',
]


class ImportModulePass(PassBase):

    __slots__ = [
        '_solvers',
        '_import_vector',
    ]

    @property
    def import_vector(self):
        return self._import_vector

    def __init__(self):
        # type: (None) -> None
        super(ImportModulePass, self).__init__()
        self._solvers = []
        self._solvers.append(ImportModuleTransformer)

    def run_pass(self, _cnode):
        solver1 = self._solvers[0]()
        _cnode.ast = solver1.visit(_cnode.ast)
        ast.fix_missing_locations(_cnode.ast)
        _modules = solver1.modules
        for _module_name, _module_src in _modules.items():
            _cleaf = CodeNodeLeaf(_cnode)
            _cleaf.src = _module_src
            _cleaf.src_file = _module_name
            _cnode.add_code_node(_cleaf)

        return _cnode
