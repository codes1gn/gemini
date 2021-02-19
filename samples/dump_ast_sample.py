import tensorflow.compat.v1 as tf

from gemini.gemini_compiler import *
from gemini.utils import *
from gemini.transformer import *

#mul_1 = tf.multiply(add_1, sub_1)


def model(input1, input2):
    add_1 = tf.add(input1, input2)
    sub_1 = input1 - input2
    mul_1 = tf.add_n([add_1, sub_1, input1, input2])
    return mul_1


try:
    compiler = GeminiCompiler()
    compiler.parse(model)
    print('dump with ast.dump\n')
    print(compiler.dump())
    print('----------------------\n')

    # TODO add to test, test visual functionality
    # wrap it wilogging_util functions., not use env vars
    # _f = ast_to_dot(f, 'try')

    import astunparse
    import inspect
    # print(astunparse.dump(ast.parse(inspect.getsource(model))))
    print('\nbefore dump')
    print(compiler.dump())
    compiler.apply_transformer(ShardingLeastDimTransformer())
    print('\nafter dump')
    print(compiler.dump())

    # test run with src code
    compiler.run(globals(), use_ast=False)
    print(model)
    _a = tf.constant(1, shape=[2, 2])
    _b = tf.constant(2, shape=[2, 2])
    logits = model(_a, _b)
    with tf.Session() as sess:
        _ = sess.run(logits)
        print(_)

    # test run with ast
    compiler.run(globals(), use_ast=True)
    print(model)
    _a = tf.constant(4, shape=[2, 2])
    _b = tf.constant(5, shape=[2, 2])
    logits = model(_a, _b)
    with tf.Session() as sess:
        _ = sess.run(logits)
        print(_)

    # TODO test round_trip


finally:
    # print(fe.ir_module.to_asm(debug_info=True))
    print("")
