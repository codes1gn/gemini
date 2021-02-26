
from gemini.utils import *
from .code_node_base import CodeNodeBase


class CodeNodeRoot(CodeNodeBase):

    __slots__ = [
        '_is_root',
        '_parent',
        '_src',
        '_ast',
        '_env',
    ]


    def __init__(self):
        # root node have no parent, if and only if
        super(CodeNodeRoot, self).__init__(None)

    def execute_module(self, module_name):
        print('root execute_import on module {}'.format(module_name))
        return

    def execute(self):
        print('root execute')
        return

