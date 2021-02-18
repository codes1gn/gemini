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
    print(compiler.dump())
    # exec(src_code, globals())
    # # exec(src_code) in globals()
    # print('global ', globals().keys())
    # print('local ', locals().keys())


if __name__ == '__main__':
    main()
