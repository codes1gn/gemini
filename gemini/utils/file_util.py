import os
import ast

__all__ = [
    'read_src',
    'dump_to_file',
]


def read_src(filename):
    # type: (basestring) -> basestring
    with open(filename, 'r') as fp:
        return fp.read()


def dump_to_file(filename, text):
    # type: (basestring, basestring) -> None
    os.system('mkdir -p dump_ast')
    with open(os.path.join("./dump_ast", filename), 'w') as fp:
        fp.write(text)
    return
