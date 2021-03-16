
import ast

from gemini.utils import *

from ..transformer.bert_device_placer_transformer import BertDevicePlacerTransformer
from .pass_base import PassBase

__all__ = [
    'PipelineDeviceStagePass',
]


class PipelineDeviceStagePass(PassBase):

    __slots__ = [
        '_solvers',
    ]

    def __init__(self):
        # type: (None) -> None
        super(PipelineDeviceStagePass, self).__init__()
        self._solvers = []
        self._solvers.append(BertDevicePlacerTransformer)

    def run_pass(self, _cnode):
        solver1 = self._solvers[0]()

        _cnode.ast = solver1.visit(_cnode.ast)
        ast.fix_missing_locations(_cnode.ast)

        if _cnode._has_sub_nodes():
            for _sub_cnode in _cnode.sub_code_nodes:
                _sub_cnode.ast = solver1.visit(_sub_cnode.ast)
                ast.fix_missing_locations(_sub_cnode.ast)

        _cnode.dump(pretty=True, prefix='bert_device_placer')
        return _cnode
