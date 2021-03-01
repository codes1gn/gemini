from gemini.utils import *

from .code_node_base import CodeNodeBase


class CodeNodeLeaf(CodeNodeBase):

    __slots__ = [
        '_is_root',
        '_parent',
        '_src',
        '_src_file',
        '_ast',
        '_env',
        '_sub_code_nodes',
    ]

    def __init__(self, parent_node):
        # root node have no parent, if and only if
        super(CodeNodeLeaf, self).__init__(parent_node)

    def parse_modules(self):
        print('leaf execute_import on module {}'.format(self.src_file))
        return

    def execute(self):
        print('leaf execute')
        return
