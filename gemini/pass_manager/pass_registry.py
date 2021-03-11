
from gemini.utils import *

from .passes import *


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
        register_pass(ImportModulePass)
        register_pass(MatmulShardingPass)
        register_pass(PluginImportFixPass)
        register_pass(PluginTransposePass)
        register_pass(PluginReshapePass)
        register_pass(PluginDropoutPass)
        register_pass(PluginLayerNormPass)
        register_pass(PluginGatherPass)
        register_pass(PluginMatmulPass)
        register_pass(PluginMultiplyPass)
        register_pass(PluginSoftmaxPass)
        register_pass(PluginDensePass)
        # register_pass(PluginPass)

        return _pass_table
