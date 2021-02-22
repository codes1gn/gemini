
from gemini.utils import *

from .pass_registry import PassRegistry


class PassManagerBase(object):

    def __init__(self):
        self.pass_vector = {}

    def add_pass(self, pass_class):
        id = PassRegistry.pass_table[pass_class.__name__]
        assert isinstance(id, basestring)
        self.pass_vector[id] = pass_class
        return

    def register_passes(self):
        print('pass_manager_base::register_passes dummy method')
        pass

    def schedule_passes(self):
        # TODO(albert) keep dummy for now
        print('pass_manager_base::schedule_passes dummy method')
        # return an ordered id list
        return ['0']

    def run_pass(self, pass_class, ast_tree):
        new_ast_tree = pass_class().run_ast(ast_tree)
        return new_ast_tree

    def run(self, compiler):
        order_list = self.schedule_passes()
        for idx in order_list:
            pass_class = self.pass_vector[idx]
            # lazy_load pass_obj
            compiler.ast = self.run_pass(pass_class, compiler.ast)
        pass
