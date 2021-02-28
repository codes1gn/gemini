
from gemini.utils import *
from gemini.pass_manager.import_module_pass_manager import *

from .code_node_base import CodeNodeBase

class CodeNodeRoot(CodeNodeBase):

    __slots__ = [
        '_is_root',
        '_parent',
        '_src',
        '_src_file',
        '_ast',
        '_env',
        '_sub_code_nodes',
    ]


    def __init__(self):
        # root node have no parent, if and only if
        super(CodeNodeRoot, self).__init__(None)

    def execute(self):
        return

