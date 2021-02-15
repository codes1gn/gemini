#!/bin/sh

path=`dirname $0`'/..'

PATH_BAK=$PYTHONPATH

export PYTHONPATH=$PYTHONPATH:$path && export DEBUG_MODE=false && \
  python $path/tests/test_compiler_dump.py

unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

