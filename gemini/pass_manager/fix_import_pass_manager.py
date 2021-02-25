
from gemini.utils import *

from .pass_manager_base import PassManagerBase
from .passes.fix_import_pass import FixImportPass

__all__ = [
    'FixImportPassManager',
]


class FixImportPassManager(PassManagerBase):

    def register_passes(self):
        print('dummy FixImportPassManager register_passes')
        self.add_pass(FixImportPass)
        return
