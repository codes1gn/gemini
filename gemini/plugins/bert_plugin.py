import tensorflow as tf
import functools
import inspect
import sys

_sharding_size = 2
_dense_sharding_switch = True

__all__ = [
    'dense',
    'matmul',
    'reshape',
    'multiply',
    'transpose',
]



if sys.version_info[0:2] >= (3, 4):  # Python v3.4+?
    wraps = functools.wraps  # built-in has __wrapped__ attribute
else:
    def wraps(wrapped, assigned=functools.WRAPPER_ASSIGNMENTS,
              updated=functools.WRAPPER_UPDATES):
        def wrapper(f):
            f = functools.wraps(wrapped, assigned, updated)(f)
            f.__wrapped__ = wrapped  # set attribute missing in earlier versions
            return f
        return wrapper


class MonadicTensor:

    def __init__(self, value):
        if isinstance(value, list) or isinstance(value, tf.Tensor):
            self.value = value
        elif isinstance(value, MonadicTensor):
            self.value = value.get()
        else:
            assert 0, 'got undefined type {}'.format(type(value))

    def get(self):
        return self.value

    def __str__(self):
        return 'MonadicTensor(' + "\n".join(map(str, self.value)) + ')'

    def __or__(self, f):
        return self.bind(f)
    
    # def __add__(self, rhs):
    #     if not isinstance(other, self.__class__):
    #         if not isinstance(other, (list, tuple)):
    #             tmp = np.ones(self._shape) * other
    #             rhs = self.__class__(list(tmp))
    #         else:
    #             rhs = self.__class__(other)
    #     else:
    #         rhs = other
    #     return self.__class__(list(map(operator.add, self._buffer, rhs._buffer)))

    def bind(self, *args, **kwargs):
        f = args[0]
        if isinstance(self.value, tf.Tensor):
            result = f(self.value, *args[1:], **kwargs)
            return MonadicTensor(result)
        elif isinstance(self.value, list):
            # FIXME, partial appends new args, not work for pending for first undefined args
            # result = list(map(functools.partial(f, *args[1:], **kwargs), self.value))
            # legacy code, use lambda function
            result = list(map(lambda inp: f(inp, *args[1:], **kwargs), self.value))
            return MonadicTensor(result)
        else:
            assert 0, 'got undefined type'

    def bind_bin_op(self, *args, **kwargs):
        f = args[0]
        rhs_operand = args[1]
        assert isinstance(self.value, type(rhs_operand.get())), \
            'bin op found unmatched lhs {} and rhs {}'.format(type(self.value), type(rhs_operand))
        if isinstance(self.value, tf.Tensor):
            result = f(self.value, rhs_operand.get(), *args[1:], **kwargs)
            return MonadicTensor(result)
        elif isinstance(self.value, list):
            # FIXME, partial appends new args, not work for pending for first undefined args
            # result = list(map(functools.partial(f, *args[1:], **kwargs), self.value))
            # legacy code, use lambda function
            result = list(map(lambda lhs, rhs: f(lhs, rhs, *args[2:], **kwargs), self.value, rhs_operand.get()))
            return MonadicTensor(result)
        else:
            assert 0, 'got undefined type'

def bind_unary_op(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # FIXME assump the input tensor is the first positional arguments in all tf.op design
        m_tensor = MonadicTensor(args[0])
        result = m_tensor.bind(f, *args[1:], **kwargs)
        return result.get()
    return wrapper

def bind_binary_op(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # FIXME assump the input tensor is the first positional arguments in all tf.op design
        lhs_tensor = MonadicTensor(args[0])
        rhs_tensor = MonadicTensor(args[1])
        result = lhs_tensor.bind_bin_op(f, rhs_tensor, *args[2:], **kwargs)
        return result.get()
    return wrapper


@bind_unary_op
def multiply(*args, **kwargs):
    return tf.multiply(*args, **kwargs)

@bind_unary_op
def transpose(*args, **kwargs):
    return tf.transpose(*args, **kwargs)

@bind_binary_op
def matmul(*args, **kwargs):
    return tf.matmul(*args, **kwargs)

# @enable_binary_op
# def all_reduce(*args, **kwargs):
#     return 0.5*tf.add(*args, **kwargs)

def all_reduce(lhs_symbol):
    if (isinstance(lhs_symbol, list) or isinstance(lhs_symbol, tuple)) and \
            isinstance(lhs_symbol[0], tf.Tensor):
        _ret = tf.add_n(lhs_symbol)
        assert isinstance(_ret, tf.Tensor)
        return _ret

    elif isinstance(lhs_symbol, tf.Tensor):
        return lhs_symbol

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(
            type(input_symbol))





def dense(*args, **kwargs):
    if not _dense_sharding_switch:
        # do not shard weights
        if isinstance(args[0], tf.Tensor):
            return tf.layers.dense(*args, **kwargs)
        elif (isinstance(args[0], tuple) or isinstance(args[0], list)) \
                and isinstance(args[0][0], tf.Tensor):
            # TODO wrap this
            _ret = []
            _rename_idx = 0
            _name_base = kwargs['name'] + "_"
            for _input_tensor in args[0]:
                kwargs['name'] = _name_base + str(_rename_idx)
                _rename_idx += 1
                print(kwargs['name'])
                _ret.append(tf.layers.dense(
                    _input_tensor, *args[1:], **kwargs))
            assert isinstance(_ret, list)
            return _ret

    elif _dense_sharding_switch:
        sharded_shape = args[1] // _sharding_size
        if isinstance(args[0], tf.Tensor):
            _ret = []
            _rename_idx = 0
            _name_base = kwargs['name'] + "_"
            for _idx in range(_sharding_size):
                kwargs['name'] = _name_base + str(_rename_idx)
                _rename_idx += 1
                _ret.append(tf.layers.dense(
                    args[0], sharded_shape, *args[2:], **kwargs))
            assert isinstance(_ret, list)
            return _ret

        elif (isinstance(args[0], tuple) or isinstance(args[0], list)) \
                and isinstance(args[0][0], tf.Tensor):
            _ret = []
            _rename_idx = 0
            _name_base = kwargs['name'] + "_"
            for _i_tensor in args[0]:
                kwargs['name'] = _name_base + str(_rename_idx)
                _rename_idx += 1
                _ret.append(tf.layers.dense(
                    _i_tensor, sharded_shape, *args[2:], **kwargs))
            # replace by reduce
            _ret_tensor = tf.add_n(_ret)
            assert isinstance(_ret_tensor, tf.Tensor)
            return _ret_tensor

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(
            type(input_symbol))


def reshape(*args, **kwargs):
    input_symbol = args[0]
    if (isinstance(input_symbol, list) or isinstance(input_symbol, tuple)) and \
            isinstance(input_symbol[0], tf.Tensor):
        _ret = []
        old_shape = args[1]
        new_shape = old_shape
        new_shape[-1] = old_shape[-1] // _sharding_size
        for _input_tensor in input_symbol:
            _ret.append(tf.reshape(_input_tensor,
                                   new_shape, *args[2:], **kwargs))
        assert isinstance(_ret, list)
        return _ret

    elif isinstance(input_symbol, tf.Tensor):
        return tf.reshape(*args, **kwargs)

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(
            type(input_symbol))
            


# legacy code
# def multiply(*args, **kwargs):
#     lhs_symbol = args[0]
# 
#     if (isinstance(lhs_symbol, list) or isinstance(lhs_symbol, tuple)) and \
#             isinstance(lhs_symbol[0], tf.Tensor):
#         _ret = []
#         for _idx in range(len(lhs_symbol)):
#             lhs_operand = lhs_symbol[_idx]
#             # rhs_operand = rhs_symbol[_idx]
#             _ret.append(tf.multiply(
#                 lhs_operand,
#                 # rhs_operand,
#                 *args[1:],
#                 **kwargs)
#             )
#         assert isinstance(_ret, list)
#         return _ret
# 
#     elif isinstance(lhs_symbol, tf.Tensor):
#         return tf.multiply(*args, **kwargs)
# 
#     else:
#         assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(
#             type(input_symbol))


# legacy transpose
# def transpose(*args, **kwargs):
#     input_symbol = args[0]
#     if (isinstance(input_symbol, list) or isinstance(input_symbol, tuple)) and \
#             isinstance(input_symbol[0], tf.Tensor):
#         _ret = []
#         for _input_tensor in input_symbol:
#             _ret.append(tf.transpose(_input_tensor, *args[1:], **kwargs))
#         assert isinstance(_ret, list)
#         return _ret
#
#     elif isinstance(input_symbol, tf.Tensor):
#         return tf.transpose(*args, **kwargs)
#
#     else:
#         assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(
#             type(input_symbol))

# legacy code: matmul
# def matmul(*args, **kwargs):
#     lhs_symbol = args[0]
#     rhs_symbol = args[1]
#     assert isinstance(lhs_symbol, type(rhs_symbol))
#     if (isinstance(lhs_symbol, list) or isinstance(lhs_symbol, tuple)) and \
#             isinstance(lhs_symbol[0], tf.Tensor):
#         _ret = []
#         for _idx in range(len(lhs_symbol)):
#             lhs_operand = lhs_symbol[_idx]
#             rhs_operand = rhs_symbol[_idx]
#             _ret.append(tf.matmul(
#                 lhs_operand,
#                 rhs_operand,
#                 *args[2:],
#                 **kwargs)
#             )
#         assert isinstance(_ret, list)
#         return _ret
# 
#     elif isinstance(lhs_symbol, tf.Tensor):
#         return tf.matmul(*args, **kwargs)
# 
#     else:
#         assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(
#             type(input_symbol))