from enum import Enum


__all__ = [
    'Configuration',
    'Mode',
]


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
    ]

    def __init__(self, v_mode=Mode.SHARDING, v_sharding_size=4, v_sharding_axis=-1):
        # default behaviour, follow megatron setting
        self._mode = v_mode
        self._sharding_size = v_sharding_size
        self._sharding_axis = v_sharding_axis

    def load_config(self, config_file):
        # TODO
        assert 0, 'Not Implemented'

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in Mode, 'mode should select in {}'.format(str(list(Mode)))
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
    def sharding_axis(self):
        return self._sharding_axis

    @sharding_axis.setter
    def sharding_axis(self, value):
        # TODO requires sanity check
        assert value in [0, -1], 'sharding_axis should be 0 or -1'
        self._sharding_axis = value

    def __str__(self):
        _str = 'ModelParallelConfiguration\n' + \
            '---- mode = {}\n'.format(self.mode) + \
            '---- sharding_size = {}\n'.format(self.sharding_size) + \
            '---- sharding_axis = {}\n'.format(self.sharding_axis)
        return _str



if __name__ == '__main__':
    config = Configuration()
    print(config.sharding_axis)
    print(config)
    config.sharding_axis=3
