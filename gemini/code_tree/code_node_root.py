
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

    def parse_modules(self):
        _pass_manager = ReadImportPassManager()
        _pass_manager.register_passes()
        _pass_manager.run(self)
        if len(self.sub_code_nodes) > 0:
            for _sub_cnode in self.sub_code_nodes:
                _sub_cnode.parse_modules()

        return

    def execute(self):
        print('root execute')
        return

