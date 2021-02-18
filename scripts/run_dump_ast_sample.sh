#!/bin/sh

path=`dirname $0`'/..'
rpath=`realpath $path`

PATH_BAK=$PYTHONPATH

export PYTHONPATH=$PYTHONPATH:$rpath && export DEBUG_MODE=false && python $rpath/samples/dump_ast_sample.py > log
unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

vim log

