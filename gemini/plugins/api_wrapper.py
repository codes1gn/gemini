import tensorflow as tf
from gemini.utils import *
from .monad import MonadicTensor


def reduce_unary_op(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # FIXME assump the input tensor is the first positional arguments in
        # all tf.op design
        m_tensor = MonadicTensor(args[0])
        result = m_tensor.reduce(f, *args[1:], **kwargs)
        return result.get()
    return wrapper

# def infer_shape(f):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         # FIXME assump the input tensor is the first positional arguments in
#         # all tf.op design
#         if kwargs.__contains__('shape'):
#             result = m_tensor.bind(f, *args, **kwargs)
#         else:
#             old_shape = args[1]
#             return f(*args, **kwargs)
#     return wrapper


def bind_unary_op(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # FIXME assump the input tensor is the first positional arguments in
        # all tf.op design
        if kwargs.__contains__('inputs'):
            m_tensor = MonadicTensor(kwargs['inputs'])
            result = m_tensor.bind(f, *args, **kwargs)
        else:
            if isinstance(args[0], float):
                _ = tf.constant(args[0])
            else:
                _ = args[0]
            m_tensor = MonadicTensor(_)
            result = m_tensor.bind(f, *args[1:], **kwargs)
        return result.get()
    return wrapper


def bind_binary_op(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # FIXME assump the input tensor is the first positional arguments in
        # all tf.op design
        lhs_tensor = MonadicTensor(args[0])
        rhs_tensor = MonadicTensor(args[1])
        result = lhs_tensor.bind_bin_op(f, rhs_tensor, *args[2:], **kwargs)
        return result.get()
    return wrapper
