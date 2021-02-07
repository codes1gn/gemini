from gemini.compiler import *
import tensorflow.compat.v1 as tf

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
  f = GeminiCompiler().parse_function(model)
finally:
  # print(fe.ir_module.to_asm(debug_info=True))
  print("")


