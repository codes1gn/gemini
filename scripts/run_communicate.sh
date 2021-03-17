#!/bin/bash
path=`dirname $0`'/..'
rpath=`realpath $path`

PATH_BAK=$PYTHONPATH

export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MIN_VLOG_LEVEL=2
export ENABLE_NEW_EXECUTABLE=true

export PYTHONPATH=$PYTHONPATH:$rpath && export DEBUG_MODE=false && \
  gdb -ex r --args python $rpath/tests/test_singleP2P.py

unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

