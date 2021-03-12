import tensorflow as tf
from functools import reduce
import copy

from gemini.plugins.api_wrapper import \
    reduce_unary_op, \
    bind_unary_op, \
    bind_binary_op
from gemini.utils import *


config = Configuration()
# define global configuration
_sharding_size = config.sharding_size
_dense_sharding_switch = True if _sharding_size > 1 and config.mode is Mode.SHARDING else False

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
    return tf.contrib.layers.layer_norm(*args, **kwargs)


@bind_unary_op
def dropout(*args, **kwargs):
    return tf.nn.dropout(*args, **kwargs)


@bind_unary_op
def gather(*args, **kwargs):
    return tf.gather(*args, **kwargs)


@bind_binary_op
def matmul(*args, **kwargs):
    return tf.matmul(*args, **kwargs)


@reduce_unary_op
def all_reduce(*args, **kwargs):
    return 1 / _sharding_size * tf.add(*args, **kwargs) \
        if _dense_sharding_switch \
        else tf.add(*args, **kwargs)


def all_gather(*args, **kwargs):
    kwargs['axis'] = -1
    return tf.concat(*args, **kwargs)


# TODO supporting -1 in shape
def _infer_new_shape(_tensor, old_shape):
    new_shape = copy.deepcopy(old_shape)
    # FIXME currently, assume only consider reshape parallel case with sharded
    # last dimension.
    # assuming given shape has no -1
    # assert -1 not in old_shape, 'has unknown shape in gemini.reshape shape_infer'
    new_shape[-1] = -1
    return new_shape


@bind_unary_op
def reshape(*args, **kwargs):
    new_shape = _infer_new_shape(args[0], args[1])
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

if __name__ == '__main__':
    # FIXME avoid both -1 in shapelist
    data = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    # new_data = reshape(data, [-1, 4])
    # new_datat = reshape(data, [4, -1])
    # print(new_data)
    # print(new_datat)
    new_data2 = reshape([data, data, data, data], [-1, 4])
    print(new_data2)
    new_data3 = reshape([data, data, data, data], [4, -1])
    print(new_data3)
    new_data4 = reshape([data, data, data, data], [4, -1, 8])
    print(new_data4)
    assert 0
