import argparse
import sys

# import tensorflow.compat.v1 as tf

from gemini.gemini_compiler import *
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

    dump_to_file('dump_0_src.ast', compiler.dump())
    dump_to_file('dump_0_src.src', compiler.dump_src())
    compiler.apply_transformer(SetParentTransformer())
    dump_to_file('dump_1_setparent.ast', compiler.dump())
    dump_to_file('dump_1_setparent.src', compiler.dump_src())

    pass1 = ShardingLeastDimTransformer(sharding_size=2)
    compiler.apply_transformer(pass1)
    dump_to_file('dump_2_shardingleastdim.ast', compiler.dump())
    dump_to_file('dump_2_shardingleastdim.src', compiler.dump_src())

    use_ast = False
    if not use_ast:
        try:
            compiler.run(globals(), use_ast=False)
            print('try run src success')
        except Exception:
            print('try run src fail')
    else:
        try:
            compiler.run(globals(), use_ast=True)
            print('try run ast success')
        except Exception:
            print('try run ast fail')


if __name__ == '__main__':
    main()
