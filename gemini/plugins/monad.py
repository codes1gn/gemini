import tensorflow as tf
from functools import reduce
import operator
from gemini.utils import *

config = Configuration()

class MonadicTensor:

    def __init__(self, value):
        if isinstance(value, list) or isinstance(value, tf.Tensor):
            self.value = value
        elif isinstance(value, self.__class__):
            self.value = value.get()
        else:
            assert 0, 'got undefined type {}'.format(type(value))

    def get(self):
        return self.value

    def __str__(self):
        if isinstance(self.value, tf.Tensor):
            return 'MonadicTensor has one Tensor:\n' + str(self.value)
        else:
            return 'MonadicTensor(' + "\n".join(map(str, self.value)) + ')'

    def __or__(self, f):
        return self.bind(f)

    def __add__(self, rhs):
        # other type, allowed: self.__class__/list,tuple/tf.Tensor
        if isinstance(rhs, self.__class__):
            # not rhs is monadic tensor as well
            if isinstance(self.value, tf.Tensor) and isinstance(
                    rhs.get(), tf.Tensor):
                return self.value + rhs.get()
            elif (isinstance(self.value, list) or isinstance(self.value, tuple)) \
                    and (isinstance(rhs.get(), list) or isinstance(rhs.get(), tuple)):
                assert len(self.value) == len(rhs.get()), 'MonadicTensor.__add__ error, with case 1' + \
                    'self.value:MonadicTensor and rhs:MonadicTensor has mismatch length'
                return self.__class__(
                    list(map(lambda l, r: l + r, self.value, rhs.get())))
            elif (isinstance(self.value, list) or isinstance(self.value, tuple)) \
                    and isinstance(rhs.get(), tf.Tensor):
                return self.__class__(
                    list(map(lambda l: l + rhs.get(), self.value)))
            elif (isinstance(rhs.get(), list) or isinstance(rhs.get(), tuple)) \
                    and isinstance(self.value, tf.Tensor):
                return self.__class__(
                    list(map(lambda r: self.value + r, rhs.get())))
            else:
                assert 0, 'MonadicTensor.__add__ error, with case 2'
        elif isinstance(rhs, list) or isinstance(rhs, tuple):
            if isinstance(self.value, tf.Tensor):
                return self.__class__(list(map(lambda r: self.value + r, rhs)))
            elif isinstance(self.value, list) or isinstance(self.value, tuple):
                assert len(self.value) == len(rhs), 'MonadicTensor.__add__ error, with case 3' + \
                    'self.value:MonadicTensor and rhs:list has mismatch length'
                return self.__class__(
                    list(map(lambda l, r: l + r, self.value, rhs)))
            else:
                assert 0, 'MonadicTensor.__add__ error, with case 4'
        elif isinstance(rhs, tf.Tensor):
            if isinstance(self.value, tf.Tensor):
                return self.value + rhs
            elif isinstance(self.value, list) or isinstance(self.value, tuple):
                return self.__class__(list(map(lambda l: l + rhs, self.value)))
            else:
                assert 0, 'MonadicTensor.__add__ error, with case 5'
        else:
            assert 0, 'MonadicTensor.__add__ error, with case 6'

    def reduce(self, *args, **kwargs):
        f = args[0]
        if isinstance(self.value, tf.Tensor):
            return self
        elif isinstance(self.value, list):
            # FIXME, partial appends new args, not work for pending for first undefined args
            # result = list(map(functools.partial(f, *args[1:], **kwargs), self.value))
            # legacy code, use lambda function
            result = reduce(f, self.value)
            assert isinstance(result, tf.Tensor), 'should be tf.Tensor'
            return self.__class__(result)
        else:
            assert 0, 'got undefined type'

    def bind(self, *args, **kwargs):
        f = args[0]
        if isinstance(self.value, tf.Tensor):
            if not kwargs.__contains__('inputs'):
                result = f(self.value, *args[1:], **kwargs)
            else:
                kwargs['inputs'] = self.value
                result = f(*args[1:], **kwargs)
            return self.__class__(result)
        elif isinstance(self.value, list):
            # FIXME, partial appends new args, not work for pending for first undefined args
            # result = list(map(functools.partial(f, *args[1:], **kwargs), self.value))
            # legacy code, use lambda function
            if kwargs.__contains__('name'):
                # avoid REUSE of vars, need to updates tensor/var/op name
                _name_base = kwargs.pop('name') + "_shard_"
                _name_candidates = []
                for idx in range(len(self.value)):
                    _name_candidates.append(_name_base + str(idx))
                # FIXME not containing kwargs handling inputs
                def _run_shard(idx, _one_name, _value):
                    _key = 'shard_' + str(idx)
                    _device_str = config.device_mapping[_key]
                    with tf.device(_device_str):
                        return f(_value, *args[1:], name=_one_name, **kwargs)

                result = []
                for idx in range(len(self.value)):
                    result.append(_run_shard(idx, _name_candidates[idx], self.value[idx]))

                # result = list(map(_run_shard, range(len(self.value)), _name_candidates, self.value))
                # result = list(map(lambda idx, inp, _one_name: ( \
                #     with tf.device(config.device_mapping['shard_' + str(idx)]):, \
                #         f(inp, *args[1:], name=_one_name, **kwargs),
                # ), range(len(self.value)), self.value, _name_candidates))
            else:
                def _run_shard(idx):
                    _key = 'shard_' + str(idx)
                    _device_str = config.device_mapping[_key]
                    with tf.device(_device_str):
                        return f(self.value[idx], *args[1:], **kwargs)

                result = []
                for idx in range(len(self.value)):
                    result.append(_run_shard(idx))
                # result = list(map(_run_shard, range(len(self.value))))
                # result = list(map(lambda _idx: f(self.value[_idx], *args[1:], **kwargs), range(len(self.value))))
            return self.__class__(result)
        else:
            assert 0, 'got undefined type'

    def bind_bin_op(self, *args, **kwargs):
        f = args[0]
        rhs_operand = args[1]
        assert isinstance(self.value, type(rhs_operand.get())), \
            'bin op found unmatched lhs {} and rhs {}'.format(
                type(self.value), type(rhs_operand))
        if isinstance(self.value, tf.Tensor):
            result = f(self.value, rhs_operand.get(), *args[2:], **kwargs)
            return self.__class__(result)
        elif isinstance(self.value, list):
            # FIXME, partial appends new args, not work for pending for first undefined args
            # result = list(map(functools.partial(f, *args[1:], **kwargs), self.value))
            result = list(map(lambda lhs, rhs: f(lhs, rhs, *
                                                 args[2:], **kwargs), self.value, rhs_operand.get()))
            return self.__class__(result)
        else:
            assert 0, 'got undefined type'
