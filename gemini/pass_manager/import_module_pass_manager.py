
from gemini.utils import *

from .pass_manager_base import PassManagerBase
from .passes.import_module_pass import ImportModulePass

__all__ = [
    'ImportModulePassManager',
]


class ImportModulePassManager(PassManagerBase):

    def register_passes(self):
        print('import_module_pass_manager::register_passes')
        self.add_pass(ImportModulePass)
        return
