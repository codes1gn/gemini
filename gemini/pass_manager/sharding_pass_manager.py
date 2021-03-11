
from gemini.utils import *

from .pass_manager_base import PassManagerBase
# from .passes.matmul_sharding_pass import MatmulShardingPass
from .passes.plugin_import_fix_pass import PluginImportFixPass
from .passes.plugin_transpose_pass import PluginTransposePass
from .passes.plugin_reshape_pass import PluginReshapePass
from .passes.plugin_dropout_pass import PluginDropoutPass
from .passes.plugin_layer_norm_pass import PluginLayerNormPass
from .passes.plugin_matmul_pass import PluginMatmulPass
from .passes.plugin_gather_pass import PluginGatherPass
from .passes.plugin_softmax_pass import PluginSoftmaxPass
from .passes.plugin_dense_pass import PluginDensePass

__all__ = [
    'ShardingPassManager',
]


class ShardingPassManager(PassManagerBase):

    def register_passes(self):
        print('sharding_pass_manager::register_passes')
        # self.add_pass(MatmulShardingPass)
        self.add_pass(PluginImportFixPass)
        self.add_pass(PluginTransposePass)
        self.add_pass(PluginReshapePass)
        self.add_pass(PluginDropoutPass)
        # self.add_pass(PluginLayerNormPass)
        # self.add_pass(PluginMatmulPass)
        self.add_pass(PluginGatherPass)
        self.add_pass(PluginSoftmaxPass)
        # self.add_pass(PluginDensePass)

        return
