
from gemini.utils import *

from .pass_manager_base import PassManagerBase
from .passes.matmul_sharding_pass import MatmulShardingPass 

__all__ = [
    'ShardingPassManager',
]


class ShardingPassManager(PassManagerBase):

    def register_passes(self):
        self.add_pass(MatmulShardingPass)
        return
