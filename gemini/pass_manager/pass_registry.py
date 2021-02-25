
from gemini.utils import *

from .passes.matmul_sharding_pass import MatmulShardingPass
from .passes.fix_import_pass import FixImportPass


class PassRegistry(object):

    @memoized_classproperty
    def pass_table(cls):

        # register passes into registry
        # will only run once
        # TODO(albert) add more passes
        _pass_table = {}

        def register_pass(pass_class):
            _pass_table[pass_class.__name__] = str(register_pass.id_cnt)
            register_pass.id_cnt += 1

        # register passes
        register_pass.id_cnt = 0
        register_pass(MatmulShardingPass)
        register_pass(FixImportPass)

        return _pass_table
