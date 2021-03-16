
from gemini.utils import *

from .pass_manager_base import PassManagerBase
from .passes.pipeline_device_stage_pass import PipelineDeviceStagePass

__all__ = [
    'PipelinePassManager',
]


class PipelinePassManager(PassManagerBase):

    def register_passes(self):
        print('pipeline_pass_manager::register_passes')
        self.add_pass(PipelineDeviceStagePass)
        return
