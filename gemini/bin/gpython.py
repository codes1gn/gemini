import argparse
import sys

# import tensorflow.compat.v1 as tf

from gemini.compiler import *
from gemini.utils import *
from gemini.transformer import *


def _get_compiler(filename, arguments):
    compiler = GeminiCompiler()
    src_code = read_src(filename)
    compiler.parse(src_code, filename=filename)
    return compiler


def main(argv=sys.argv[1:]):
    print('gemini compiler entry point')
    filename = argv[0]
    arguments = argv[1:]
    compiler = _get_compiler(filename, arguments)
    dump_to_file('dump_1.ast', compiler.dump())
    compiler.apply_transformer(ShardingLeastDimTransformer())
    dump_to_file('dump_2.ast', compiler.dump())

    # lastly, run source codes
    # print('try run with src code')
    # compiler.run(globals(), use_ast=False)
    # compiler.run(globals(), use_ast=True)

    #  exec(compiler.src, globals())
    #  print(globals().keys())


if __name__ == '__main__':
    main()
