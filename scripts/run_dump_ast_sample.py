path=`dirname $0`'/..'
echo $path

PATH_BAK=$PYTHONPATH

export PYTHONPATH=$PYTHONPATH:$path && export DEBUG_MODE=false && python $path/samples/dump_ast_sample.py > log
unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

vim log
