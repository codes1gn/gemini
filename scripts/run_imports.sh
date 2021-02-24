#!/bin/bash

path=`dirname $0`'/..'
rpath=`realpath $path`

PATH_BAK=$PYTHONPATH

# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MIN_VLOG_LEVEL=2
# export TF_DUMP_GRAPH_PREFIX=./dump_graph
runner="gemini_python"

if [ "$1" = "vanilla" ]; then
  python $rpath/samples/import_entry.py
  unset PYTHONPATH
  export PYTHONPATH=$PATH_BAK
  exit 0
fi

if ! command -v $runner > /dev/null;
then
  echo "using bin/gemini_python.py"
  export PYTHONPATH=$PYTHONPATH:$rpath:$rpath"/samples" && export DEBUG_MODE=false && python $rpath/gemini/bin/gemini_python.py $rpath/samples/import_entry.py # --max_steps 5
else
  echo "using gemini_python"
  export PYTHONPATH=$PYTHONPATH:$rpath:$rpath"/samples" && export DEBUG_MODE=false && gemini_python $rpath/samples/import_entry.py # --max_steps 5
fi

unset PYTHONPATH
export PYTHONPATH=$PATH_BAK

