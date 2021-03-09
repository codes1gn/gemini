import tensorflow as tf
from functools import reduce
import operator


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
        return 'MonadicTensor(' + "\n".join(map(str, self.value)) + ')'

    def __or__(self, f):
        return self.bind(f)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            if isinstance(other, (list, tuple)):
                rhs = self.__class__(other)
            elif isinstance(other, tf.Tensor):
                _list = []
                for idx in range(len(self.value)):
                    _list.append(other)
                rhs = self.__class__(list(_list))
            else:
                assert 0, 'expect tf.Tensor or literal, but got list/tuple'
        else:
            rhs = other
        return self.__class__(list(map(operator.add, self.value, rhs.get())))

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
            result = f(self.value, *args[1:], **kwargs)
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
                result = list(
                    map(lambda inp, _one_name: f(inp, *args[1:], name=_one_name, **kwargs), self.value, _name_candidates))
            else:
                result = list(
                    map(lambda inp: f(inp, *args[1:], **kwargs), self.value))
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
