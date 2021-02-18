import argparse
import sys

import tensorflow.compat.v1 as tf

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
    print(src_code)
    if len(arguments) == 2:
        print('args are one pair')
        # exec()

if __name__ == '__main__':
    main()
