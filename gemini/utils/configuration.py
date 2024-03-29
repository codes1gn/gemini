from enum import Enum


__all__ = [
    'Configuration',
    'Mode',
    # 'get_stage_by_tensor_name',
]

_layers_to_stage = {
    0: '0',
    1: '1',
    2: '1',
    3: '2',
    4: '2',
    5: '3',
    6: '3',
    7: '3',
    8: '4',
    9: '4',
    10: '4',
    11: '5',
    12: '5',
    13: '5',
    14: '5',
    15: '6',
    16: '6',
    17: '6',
    18: '6',
    19: '7',
    20: '7',
    21: '7',
    22: '7',
    23: '7',
}


def _get_stage_by_tensor_name(_name):
    # TODO design better logics
    # FIXME hardcode logics to result the 'stage_?' from tensor names

    if 'embedding' in _name:
        return "stage_0"
    if 'encoder' in _name and 'layer_' not in _name:
        return "stage_0"
    elif 'layer_' in _name:
        # FIXME more generic
        if '/encoder' in _name:
            _layer = int(_name.split('/')[3].split('_')[-1])
        else:
            print('anchor', _name)
            _layer = int(_name.split('_')[-1])
        _stage = "stage_" + _layers_to_stage[_layer]
        return _stage
    elif 'pooler' in _name:
        return 'stage_7'
    elif 'loss' in _name:
        return 'stage_7'
    else:
        return 'fail'


class Mode(Enum):
    SHARDING = 1
    GROUPPIPE = 2
    INTERPIPE = 3
    SEQPIPE = 4
    VANILLA = 5


class Configuration(object):
    """Global config object"""
    __slots__ = [
        '_mode',
        '_sharding_size',
        '_sharding_axis',
        '_device_mapping',
        '_accum_degree',
    ]

    def __init__(self, v_mode=Mode.SHARDING, v_sharding_size=4,
                 v_sharding_axis=-1, v_accum_degree=1):
        # default behaviour, follow megatron setting
        self._mode = v_mode
        self._sharding_size = v_sharding_size
        self._sharding_axis = v_sharding_axis
        self._accum_degree = v_accum_degree
        self._device_mapping = {}
        self._device_mapping['stage_0'] = {}
        self._device_mapping['stage_1'] = {}
        self._device_mapping['stage_2'] = {}
        self._device_mapping['stage_3'] = {}
        self._device_mapping['stage_4'] = {}
        self._device_mapping['stage_5'] = {}
        self._device_mapping['stage_6'] = {}
        self._device_mapping['stage_7'] = {}

        # stage 0
        self._device_mapping['stage_0']['shard_0'] = '/device:XLA_DTU:0'
        self._device_mapping['stage_0']['shard_1'] = '/device:XLA_DTU:1'
        self._device_mapping['stage_0']['shard_2'] = '/device:XLA_DTU:2'
        self._device_mapping['stage_0']['shard_3'] = '/device:XLA_DTU:3'

        # uncomment it and comment follows if on 1 dtu machine
        # self._device_mapping['stage_1']['shard_0'] = '/device:XLA_DTU:0'
        # self._device_mapping['stage_1']['shard_1'] = '/device:XLA_DTU:1'
        # self._device_mapping['stage_1']['shard_2'] = '/device:XLA_DTU:2'
        # self._device_mapping['stage_1']['shard_3'] = '/device:XLA_DTU:3'
        # self._device_mapping['stage_2']['shard_0'] = '/device:XLA_DTU:0'
        # self._device_mapping['stage_2']['shard_1'] = '/device:XLA_DTU:1'
        # self._device_mapping['stage_2']['shard_2'] = '/device:XLA_DTU:2'
        # self._device_mapping['stage_2']['shard_3'] = '/device:XLA_DTU:3'
        # self._device_mapping['stage_3']['shard_0'] = '/device:XLA_DTU:0'
        # self._device_mapping['stage_3']['shard_1'] = '/device:XLA_DTU:1'
        # self._device_mapping['stage_3']['shard_2'] = '/device:XLA_DTU:2'
        # self._device_mapping['stage_3']['shard_3'] = '/device:XLA_DTU:3'
        # self._device_mapping['stage_4']['shard_0'] = '/device:XLA_DTU:0'
        # self._device_mapping['stage_4']['shard_1'] = '/device:XLA_DTU:1'
        # self._device_mapping['stage_4']['shard_2'] = '/device:XLA_DTU:2'
        # self._device_mapping['stage_4']['shard_3'] = '/device:XLA_DTU:3'
        # self._device_mapping['stage_5']['shard_0'] = '/device:XLA_DTU:0'
        # self._device_mapping['stage_5']['shard_1'] = '/device:XLA_DTU:1'
        # self._device_mapping['stage_5']['shard_2'] = '/device:XLA_DTU:2'
        # self._device_mapping['stage_5']['shard_3'] = '/device:XLA_DTU:3'
        # self._device_mapping['stage_6']['shard_0'] = '/device:XLA_DTU:0'
        # self._device_mapping['stage_6']['shard_1'] = '/device:XLA_DTU:1'
        # self._device_mapping['stage_6']['shard_2'] = '/device:XLA_DTU:2'
        # self._device_mapping['stage_6']['shard_3'] = '/device:XLA_DTU:3'
        # self._device_mapping['stage_7']['shard_0'] = '/device:XLA_DTU:0'
        # self._device_mapping['stage_7']['shard_1'] = '/device:XLA_DTU:1'
        # self._device_mapping['stage_7']['shard_2'] = '/device:XLA_DTU:2'
        # self._device_mapping['stage_7']['shard_3'] = '/device:XLA_DTU:3'

        # uncomment followings and comment above if on 8x dtu server
        # stage 1
        self._device_mapping['stage_1']['shard_0'] = '/device:XLA_DTU:5'
        self._device_mapping['stage_1']['shard_1'] = '/device:XLA_DTU:6'
        self._device_mapping['stage_1']['shard_2'] = '/device:XLA_DTU:7'
        self._device_mapping['stage_1']['shard_3'] = '/device:XLA_DTU:8'

        # stage 2
        self._device_mapping['stage_2']['shard_0'] = '/device:XLA_DTU:10'
        self._device_mapping['stage_2']['shard_1'] = '/device:XLA_DTU:11'
        self._device_mapping['stage_2']['shard_2'] = '/device:XLA_DTU:12'
        self._device_mapping['stage_2']['shard_3'] = '/device:XLA_DTU:13'

        # stage 3
        self._device_mapping['stage_3']['shard_0'] = '/device:XLA_DTU:15'
        self._device_mapping['stage_3']['shard_1'] = '/device:XLA_DTU:16'
        self._device_mapping['stage_3']['shard_2'] = '/device:XLA_DTU:17'
        self._device_mapping['stage_3']['shard_3'] = '/device:XLA_DTU:18'

        # stage 4
        self._device_mapping['stage_4']['shard_0'] = '/device:XLA_DTU:20'
        self._device_mapping['stage_4']['shard_1'] = '/device:XLA_DTU:21'
        self._device_mapping['stage_4']['shard_2'] = '/device:XLA_DTU:22'
        self._device_mapping['stage_4']['shard_3'] = '/device:XLA_DTU:23'

        # stage 5
        self._device_mapping['stage_5']['shard_0'] = '/device:XLA_DTU:25'
        self._device_mapping['stage_5']['shard_1'] = '/device:XLA_DTU:26'
        self._device_mapping['stage_5']['shard_2'] = '/device:XLA_DTU:27'
        self._device_mapping['stage_5']['shard_3'] = '/device:XLA_DTU:28'

        # stage 6
        self._device_mapping['stage_6']['shard_0'] = '/device:XLA_DTU:30'
        self._device_mapping['stage_6']['shard_1'] = '/device:XLA_DTU:31'
        self._device_mapping['stage_6']['shard_2'] = '/device:XLA_DTU:32'
        self._device_mapping['stage_6']['shard_3'] = '/device:XLA_DTU:33'

        # stage 7
        self._device_mapping['stage_7']['shard_0'] = '/device:XLA_DTU:35'
        self._device_mapping['stage_7']['shard_1'] = '/device:XLA_DTU:36'
        self._device_mapping['stage_7']['shard_2'] = '/device:XLA_DTU:37'
        self._device_mapping['stage_7']['shard_3'] = '/device:XLA_DTU:38'

    def get_device_by_tensor(self, _tensor, shard_idx=0):
        _stage = _get_stage_by_tensor_name(_tensor.name)
        if 'stage' not in _stage:
            return 'CPU:0'
        _device_str = self.device_mapping[_stage]['shard_{}'.format(
            str(shard_idx))]
        print('debugoo', _tensor.name, _device_str)
        return _device_str

    def get_device_by_tensor_name(self, _tensor_name, shard_idx=0):
        _stage = _get_stage_by_tensor_name(_tensor_name)
        if 'stage' not in _stage:
            return 'CPU:0'
        _device_str = self.device_mapping[_stage]['shard_{}'.format(
            str(shard_idx))]
        print('debugoo', _tensor_name, _device_str)
        return _device_str

    def load_config(self, config_file):
        # TODO
        assert 0, 'Not Implemented'

    @property
    def device_mapping(self):
        # TODO check order
        return {key: self._device_mapping[key]
                for key in sorted(self._device_mapping.keys())}

    @device_mapping.setter
    def device_mapping(self, value):
        assert isinstance(
            value, dict), 'device mapping should be dict type, but got {}'.format(
            type(value))
        self._device_mapping = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in Mode, 'mode should select in {}'.format(
            str(list(Mode)))
        self._mode = value

    @property
    def sharding_size(self):
        return self._sharding_size

    @sharding_size.setter
    def sharding_size(self, value):
        # TODO requires sanity check
        assert value in [1, 2, 4], 'sharding_axis should be 1, 2 or 4'
        self._sharding_size = value

    @property
    def accum_degree(self):
        return self._accum_degree

    @accum_degree.setter
    def accum_degree(self, value):
        # TODO requires sanity check
        assert value in [1, 2, 4, 8, 16,
                         32], 'sharding_axis should be 1, 2 or 4'
        self._accum_degree = value

    @property
    def sharding_axis(self):
        return self._sharding_axis

    @sharding_axis.setter
    def sharding_axis(self, value):
        # TODO requires sanity check
        assert value in [0, -1], 'sharding_axis should be 0 or -1'
        self._sharding_axis = value

    def __str__(self):
        def pretty(d, indent=0):
            _s = ""
            for key, value in d.items():
                _s = _s + '\t' * indent + str(key)
                if isinstance(value, dict):
                    _s = _s + pretty(value, indent + 1)
                else:
                    _s = _s + '\t' * (indent + 1) + str(value) + '\n'
            return _s

        _str = 'ModelParallelConfiguration\n' + \
            '---- mode = {}\n'.format(self.mode) + \
            '---- sharding_size = {}\n'.format(self.sharding_size) + \
            '---- sharding_axis = {}\n'.format(self.sharding_axis) + \
            '---- device_mapping = \n{}'.format(pretty(self.device_mapping))
        return _str


if __name__ == '__main__':
    # # test config
    # config = Configuration()
    # print(config.sharding_axis)
    # print(config)
    # config.sharding_axis=3

    # test stage_func
    # assert "stage_0" == get_stage_by_tensor_name("bert/embeddings/split:0"), 'fail'
    # print('pass all')
    # test new
    config = Configuration()
    print(config.get_device_by_tensor_name('embeddings'))
    print(config.get_device_by_tensor_name('layer_11'))
    print(config.get_device_by_tensor_name('gemini/bert/encoder/layer_7/haha'))
