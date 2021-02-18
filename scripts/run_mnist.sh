#!/bin/bash

path=`dirname $0`'/..'
rpath=`realpath $path`

PATH_BAK=$PYTHONPATH

# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MIN_VLOG_LEVEL=2
# export TF_DUMP_GRAPH_PREFIX=./dump_graph
export PYTHONPATH=$PYTHONPATH:$rpath && export DEBUG_MODE=false && python $rpath/gemini/bin/gpython.py $rpath/samples/mnist.py # --max_steps 5
unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

