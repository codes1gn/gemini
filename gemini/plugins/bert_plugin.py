import tensorflow as tf
from functools import reduce
import copy

from .api_wrapper import \
    reduce_unary_op, \
    bind_unary_op, \
    bind_binary_op


# define global configuration
_sharding_size = 2
_dense_sharding_switch = True

# _sharding_size = 1
# _dense_sharding_switch = False


# define visibility
__all__ = [
    'dense',
    'matmul',
    'reshape',
    'multiply',
    'transpose',
    'dropout',
    'all_reduce',
    'softmax',
]


@bind_unary_op
def multiply(*args, **kwargs):
    return tf.multiply(*args, **kwargs)


@bind_unary_op
def transpose(*args, **kwargs):
    return tf.transpose(*args, **kwargs)


@bind_unary_op
def softmax(*args, **kwargs):
    return tf.nn.softmax(*args, **kwargs)


@bind_unary_op
def layer_norm(*args, **kwargs):
    print(kwargs)
    print(args)
    return tf.contrib.layers.layer_norm(*args, **kwargs)


@bind_unary_op
def dropout(*args, **kwargs):
    return tf.nn.dropout(*args, **kwargs)


@bind_binary_op
def matmul(*args, **kwargs):
    return tf.matmul(*args, **kwargs)


@reduce_unary_op
def all_reduce(*args, **kwargs):
    return 1 / _sharding_size * tf.add(*args, **kwargs) \
        if _dense_sharding_switch \
        else tf.add(*args, **kwargs)


@bind_unary_op
def reshape(*args, **kwargs):
    # FIXME currently, assume only consider reshape parallel case with sharded
    # last dimension.
    new_shape = copy.deepcopy(args[1])
    # FIXME use -1 tmply
    # new_shape[-1] = args[1][-1] // _sharding_size
    new_shape[-1] = -1
    return tf.reshape(args[0], new_shape, *args[2:], **kwargs)


@bind_unary_op
def monadic_dense(*args, **kwargs):
    return tf.layers.dense(*args, **kwargs)


if not _dense_sharding_switch:
    def dense(*args, **kwargs):
        return monadic_dense(*args, **kwargs)
elif _dense_sharding_switch:
    def dense(*args, **kwargs):
        if isinstance(args[0], tf.Tensor):
            sharded_out_size = args[1] // _sharding_size
            input_list = [args[0], args[0]]
            return monadic_dense(
                input_list, sharded_out_size, *args[2:], **kwargs)

        elif (isinstance(args[0], tuple) or isinstance(args[0], list)) \
                and isinstance(args[0][0], tf.Tensor):
            _tmp = monadic_dense(*args, **kwargs)
            return all_reduce(_tmp)
