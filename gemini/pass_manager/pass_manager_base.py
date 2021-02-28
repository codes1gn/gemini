
from gemini.utils import *

from .pass_registry import PassRegistry


class PassManagerBase(object):

    __slots__ = [
        '_pass_vector',
        '_pass_id_list',
        '_concrete_pass',
    ]

    def __init__(self):
        self._pass_vector = {}
        self._pass_id_list = []
        self._concrete_pass = []

    @property
    def pass_vector(self):
        return self._pass_vector

    @property
    def pass_id_list(self):
        return self._pass_id_list

    @property
    def concrete_pass(self):
        return self._concrete_pass

    def add_pass(self, pass_class):
        id = PassRegistry.pass_table[pass_class.__name__]
        assert isinstance(id, basestring)
        self._pass_vector[id] = pass_class
        self._pass_id_list.append(id)
        return

    def register_passes(self):
        print('pass_manager_base::register_passes')
        pass

    def schedule_passes(self):
        # TODO(albert) keep dummy for now
        print('pass_manager_base::schedule_passes')
        # return an ordered id list
        return self._pass_id_list

    def run_pass(self, pass_class, code_node):
        cpass = pass_class()
        self._concrete_pass.append(cpass)
        code_node = cpass.run_pass(code_node)
        return

    def run(self, code_node):
        order_list = self.schedule_passes()
        for idx in order_list:
            pass_class = self._pass_vector[idx]
            # lazy_load pass_obj
            self.run_pass(pass_class, code_node)
        pass
