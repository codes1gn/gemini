
from gemini.utils import *

from .pass_manager_base import PassManagerBase
from .passes.module_trans_pass import ModuleTransPass

__all__ = [
    'ModuleTransPassManager',
]


class ModuleTransPassManager(PassManagerBase):

    def register_passes(self):
        self.add_pass(ModuleTransPass)
        return

    # need to override run_pass and run, transform on modules
    def run_pass(self, pass_class, src):
        cpass = pass_class()
        self._concrete_pass.append(cpass)
        new_src = cpass.run_src(src)
        return new_src

    def run(self, compiler):
        order_list = self.schedule_passes()
        for idx in order_list:
            pass_class = self._pass_vector[idx]
            # lazy_load pass_obj
            for idx in range(len(compiler.import_code_vector)):
                _module_src = compiler.import_code_vector[idx]
                new_src = self.run_pass(pass_class, _module_src)
                compiler.import_code_vector[idx] = new_src
        pass

