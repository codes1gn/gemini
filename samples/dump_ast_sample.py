import tensorflow.compat.v1 as tf

from gemini.compiler import *
from gemini.utils import *
from gemini.transformer import *


def model(input1, input2):
    add_1 = tf.add(input1, input2)
    sub_1 = input1 - input2
    mul_1 = tf.multiply(add_1, sub_1)
    return mul_1

# import inspect
# # source_code = model
# source_code = inspect.getsource(model)
# source_code += "logits = model(input1, input2)\n"
# print(source_code)
# _a = tf.constant(1, shape=[2, 2])
# _b = tf.constant(2, shape=[2, 2])
#
#
# # error, how to run??
# _local = {'input1':_a, 'input2':_b, 'logits': 0}
# _global = {}
# exec(source_code, _global, _local)
# logits = _local['logits']


try:
    compiler = GeminiCompiler()
    compiler.parse_function(model)
    print('dump with ast.dump\n')
    print(compiler.dump())
    print('----------------------\n')

    # TODO add to test, test visual functionality
    # wrap it wilogging_util functions., not use env vars
    # _f = ast_to_dot(f, 'try')

    # use NodeTransformer()
    # TODO add to test
    # print('dump ast_transformed with ast.dump\n')
    # print(ast.dump(code_ast_transformed))
    # print('----------------------\n')

    import astunparse
    import inspect
    # print(astunparse.dump(ast.parse(inspect.getsource(model))))
    print('before dump')
    print(astunparse.dump(compiler.ast))
    compiler.apply_transformer(ShardingLeastDimTransformer())
    print('after dump')
    print(astunparse.dump(compiler.ast))

    # TODO test pretty dump

    # TODO test round_trip


finally:
    # print(fe.ir_module.to_asm(debug_info=True))
    print("")
