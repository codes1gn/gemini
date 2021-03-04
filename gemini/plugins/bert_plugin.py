import tensorflow as tf


_sharding_size = 2
_dense_sharding_switch = True

__all__ = [
    'dense',
    'matmul',
    'reshape',
    'multiply',
    'transpose',
]


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
                _ret.append(tf.layers.dense(_input_tensor, *args[1:], **kwargs))
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
                _ret.append(tf.layers.dense(args[0], sharded_shape, *args[2:], **kwargs))
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
                _ret.append(tf.layers.dense(_i_tensor, sharded_shape, *args[2:], **kwargs))
            # replace by reduce
            _ret_tensor = tf.add_n(_ret)
            assert isinstance(_ret_tensor, tf.Tensor)
            return _ret_tensor

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(type(input_symbol))


def reshape(*args, **kwargs):
    input_symbol = args[0]
    if (isinstance(input_symbol, list) or isinstance(input_symbol, tuple)) and \
            isinstance(input_symbol[0], tf.Tensor):
        _ret = []
        old_shape = args[1]
        new_shape = old_shape
        new_shape[-1] = old_shape[-1] // _sharding_size
        for _input_tensor in input_symbol:
            _ret.append(tf.reshape(_input_tensor, new_shape, *args[2:], **kwargs))
        assert isinstance(_ret, list)
        return _ret

    elif isinstance(input_symbol, tf.Tensor):
        return tf.reshape(*args, **kwargs)

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(type(input_symbol))


def transpose(*args, **kwargs):
    input_symbol = args[0]
    if (isinstance(input_symbol, list) or isinstance(input_symbol, tuple)) and \
            isinstance(input_symbol[0], tf.Tensor):
        _ret = []
        for _input_tensor in input_symbol:
            _ret.append(tf.transpose(_input_tensor, *args[1:], **kwargs))
        assert isinstance(_ret, list)
        return _ret

    elif isinstance(input_symbol, tf.Tensor):
        return tf.transpose(*args, **kwargs)

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(type(input_symbol))

def matmul(*args, **kwargs):
    lhs_symbol = args[0]
    rhs_symbol = args[1]
    assert isinstance(lhs_symbol, type(rhs_symbol))
    if (isinstance(lhs_symbol, list) or isinstance(lhs_symbol, tuple)) and \
            isinstance(lhs_symbol[0], tf.Tensor):
        _ret = []
        for _idx in range(len(lhs_symbol)):
            lhs_operand = lhs_symbol[_idx]
            rhs_operand = rhs_symbol[_idx]
            _ret.append(tf.matmul(
                lhs_operand, 
                rhs_operand, 
                *args[2:], 
                **kwargs)
            )
        assert isinstance(_ret, list)
        return _ret

    elif isinstance(lhs_symbol, tf.Tensor):
        return tf.matmul(*args, **kwargs)

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(type(input_symbol))

def multiply(*args, **kwargs):
    lhs_symbol = args[0]
    rhs_symbol = args[1]

    if (isinstance(lhs_symbol, list) or isinstance(lhs_symbol, tuple)) and \
        isinstance(lhs_symbol[0], tf.Tensor):
        _ret = []
        for _idx in range(len(lhs_symbol)):
            lhs_operand = lhs_symbol[_idx]
            # rhs_operand = rhs_symbol[_idx]
            _ret.append(tf.multiply(
                lhs_operand, 
                # rhs_operand, 
                *args[1:], 
                **kwargs)
            )
        assert isinstance(_ret, list)
        return _ret

    elif isinstance(lhs_symbol, tf.Tensor):
        return tf.multiply(*args, **kwargs)

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(type(input_symbol))

def merge(symbol_input):
    lhs_symbol = args[0]
    rhs_symbol = args[1]
    if (isinstance(lhs_symbol, list) or isinstance(lhs_symbol, tuple)) and \
        isinstance(lhs_symbol[0], tf.Tensor):
        _ret = []
        for _idx in range(len(lhs_symbol)):
            lhs_operand = lhs_symbol[_idx]
            rhs_operand = rhs_symbol[_idx]
            _ret.append(tf.multiply(
                lhs_operand, 
                rhs_operand, 
                *args[2:], 
                **kwargs)
            )
        assert isinstance(_ret, list)
        return _ret

    elif isinstance(lhs_symbol, tf.Tensor):
        return tf.multiply(*args, **kwargs)

    else:
        assert 0, 'expected tf.Tensor or list/tuple of tf.Tensor as inputs, but got {}'.format(type(input_symbol))
