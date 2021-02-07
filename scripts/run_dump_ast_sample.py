path=`dirname $0`'/..'
echo $path

PATH_BAK=$PYTHONPATH

export PYTHONPATH=$PYTHONPATH:$path && export DEBUG_MODE=true && python $path/samples/dump_ast_sample.py
unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

