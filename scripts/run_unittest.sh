#!/bin/sh

path=`dirname $0`'/..'
rpath=`realpath $path`

PATH_BAK=$PYTHONPATH

export PYTHONPATH=$PYTHONPATH:$rpath && export DEBUG_MODE=false && \
  python $rpath/tests/test_code_node.py

# export PYTHONPATH=$PYTHONPATH:$rpath && export DEBUG_MODE=false && \
#   python $rpath/tests/test_compiler_dump.py
# 
# export PYTHONPATH=$PYTHONPATH:$rpath && export DEBUG_MODE=false && \
#   python $rpath/tests/test_mnist.py
# 
# export PYTHONPATH=$PYTHONPATH:$rpath && export DEBUG_MODE=false && \
#   python $rpath/tests/test_compiler_run.py

unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

