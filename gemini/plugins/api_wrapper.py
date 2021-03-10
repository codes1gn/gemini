
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


def bind_unary_op(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # FIXME assump the input tensor is the first positional arguments in
        # all tf.op design
        if kwargs.__contains__('inputs'):
            m_tensor = MonadicTensor(kwargs['inputs'])
            result = m_tensor.bind(f, *args, **kwargs)
        else:
            m_tensor = MonadicTensor(args[0])
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
