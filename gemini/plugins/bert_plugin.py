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


# define visibility
__all__ = [
    'dense',
    'matmul',
    'reshape',
    'multiply',
    'transpose',
]


@bind_unary_op
def multiply(*args, **kwargs):
    return tf.multiply(*args, **kwargs)

@bind_unary_op
def transpose(*args, **kwargs):
    return tf.transpose(*args, **kwargs)

@bind_binary_op
def matmul(*args, **kwargs):
    return tf.matmul(*args, **kwargs)

@reduce_unary_op
def all_reduce(*args, **kwargs):
    return 1 / _sharding_size * tf.add(*args, **kwargs)

@bind_unary_op
def reshape(*args, **kwargs):
    # FIXME currently, assume only consider reshape parallel case with sharded last dimension.
    new_shape = copy.deepcopy(args[1])
    new_shape[-1] = args[1][-1] // _sharding_size
    return tf.reshape(args[0], new_shape, *args[2:], **kwargs)

            
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
