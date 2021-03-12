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

from abc import ABC, abstractmethod

class Validator(ABC):

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass

class OneOf(Validator):

    def __init__(self, *options):
        self.options = set(options)

    def validate(self, value):
        if value not in self.options:
            raise ValueError('Expected {} to be one of {}'.format(value, self.options))

class Number(Validator):

    def __init__(self, minvalue=None, maxvalue=None):
        self.minvalue = minvalue
        self.maxvalue = maxvalue

    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError('Expected {} to be an int or float'.format(value))
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(
                'Expected {} to be at least {}'.format(value, self.minvalue)
            )
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(
                'Expected {} to be no more than {}'.format(value, self.maxvalue)
            )


class Configuration(object):
    """Global config object"""
    __slots__ = [
        '_mode',
        '_sharding_size',
        '_sharding_axis',
    ]

    v_mode = OneOf(*list(Mode))
    v_sharding_size = Number(minvalue=1, maxvalue=4)
    v_sharding_axis = OneOf(0, -1)

    def __init__(self, v_mode=Mode.SHARDING, v_sharding_size=4, v_sharding_axis=-1):
        if config_file is not None:
            # TODO parser functionality is required
            assert 0, 'Not implemented to parse from config file'
        else:
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
        assert value in Mode
        self._mode = value

    @property
    def sharding_size(self):
        return self._sharding_size

    @sharding_size.setter
    def sharding_size(self, value):
        # TODO requires sanity check
        assert value in [1, 2, 4]
        self._sharding_size = value

    @property
    def sharding_axis(self):
        return self._sharding_axis

    @sharding_axis.setter
    def sharding_axis(self, value):
        # TODO requires sanity check
        assert value in [0, -1]
        self._sharding_axis = value



if __name__ == '__main__':
    config = Configuration()
    config.finalize()
