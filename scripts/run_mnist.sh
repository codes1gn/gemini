#!/bin/bash

path=`dirname $0`'/..'
echo $path

PATH_BAK=$PYTHONPATH

# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MIN_VLOG_LEVEL=2
# export TF_DUMP_GRAPH_PREFIX=./dump_graph
export PYTHONPATH=$PYTHONPATH:$path && export DEBUG_MODE=false && python $path/gemini/bin/gpython.py # $path/samples/mnist_runner.py
unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

