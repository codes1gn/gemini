import argparse
import sys

# import tensorflow.compat.v1 as tf

from gemini.compiler import *
from gemini.utils import *
from gemini.transformer import *


def read_src(filename):
    with open(filename, 'r') as fp:
        return fp.read()


def main(argv=sys.argv[1:]):
    print('hello world')
    filename = argv[0]
    arguments = argv[1:]
    src_code = read_src(filename)
    print(filename)
    print(arguments)
    print('exec src_code')
    exec(src_code, globals())
    # exec(src_code) in globals()
    print('global ', globals().keys())
    print('local ', locals().keys())


if __name__ == '__main__':
    main()
