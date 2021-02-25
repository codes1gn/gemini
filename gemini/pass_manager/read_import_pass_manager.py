
from gemini.utils import *

from .pass_manager_base import PassManagerBase
from .passes.read_import_pass import ReadImportPass

__all__ = [
    'ReadImportPassManager',
]


class ReadImportPassManager(PassManagerBase):

    def register_passes(self):
        print('dummy ReadImportPassManager register_passes')
        self.add_pass(ReadImportPass)
        return
