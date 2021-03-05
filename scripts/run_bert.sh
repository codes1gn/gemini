#!/bin/bash

get_sha256sum() {
  cat $1 | sha256sum | head -c 64
}

string_contains () {
  [ -z "$1"  ] || { [ -z "${2##*$1*}"  ] && [ -n "$2"  ]; };
}

download_file() { 
  if [ -f "$1" ]; then
    echo "$1 exist"
    # check checksum
    if string_contains $2 `get_sha256sum $1`; then
      echo "found correct file, skip download"
      return 0
    fi
    echo "found file, but filehash failed, do download"
  else
    echo "not found file, do download"
  fi
  rm -f $1
  if string_contains "24_H-1024" $1; then
    wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
  elif string_contains "12_H-768" $1; then
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
  elif string_contains "squad" $1; then
    wget ftp://10.16.11.32/software/dataset/squad.zip
  elif string_contains "glue" $1; then
    wget ftp://10.16.11.32/software/dataset/glue.zip
  else
    echo "invalid filename specified, cannot process"
  fi
  echo "clean $1"
  echo "download $1"
  return 0
}

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath
bert_dir=$top_dir_realpath"/external/TopsModels/research/training/tensorflow/nlp/bert"
echo "bert dir = "$bert_dir

# download pretrained model, and dataset
mkdir -p $bert_dir"/pretrained_models"
mkdir -p $bert_dir"/dataset"

# download pretrained model
cd $bert_dir"/pretrained_models"
FILE=wwm_uncased_L-24_H-1024_A-16.zip
FILE_PATH=$bert_dir"/pretrained_models"$FILE
FILE_HASH="3619df03637b027f41f4bf6d3ea21d4f162129dd57b6cc223939a320046a1ec5"
download_file $FILE $FILE_HASH
FILE2=uncased_L-12_H-768_A-12.zip
FILE_PATH2=$bert_dir"/pretrained_models"$FILE2
FILE_HASH2="d15224e1e7d950fb9a8b29497ce962201dff7d27b379f5bfb4638b4a73540a04"
download_file $FILE2 $FILE_HASH2

# unzip
if [ -d "${FILE%.zip}" ]; then
  echo "dir ${FILE%.zip} exist"
  # check checksum
else
  echo "not found dir ${FILE%.zip}, do unzip"
  unzip $FILE
fi


if [ -d "${FILE2%.zip}" ]; then
  echo "dir ${FILE2%.zip} exist"
  # check checksum
else
  echo "not found dir ${FILE2%.zip}, do unzip"
  mkdir -p uncased_L-12_H-768_A-12 && cd uncased_L-12_H-768_A-12/
  unzip ../$FILE2
fi

cd -
cd $top_dir_realpath

# download dataset
cd $bert_dir"/dataset"
FILE3=squad.zip
FILE_HASH3="e49e508d867daa7c6d6365b7112c270da8d25a51dac479384083114eb6964a6e"
download_file $FILE3 $FILE_HASH3
FILE4=glue.zip
FILE_HASH4="7b3fa927167380c0ec704230d8730390a11307dbcd6fd454408cb36dfb29ab95"
download_file $FILE4 $FILE_HASH4

if [ -d "${FILE3%.zip}" ]; then
  echo "dir ${FILE3%.zip} exist"
  # check checksum
else
  echo "not found dir ${FILE3%.zip}, do unzip"
  unzip $FILE3
fi

if [ -d "glue_data" ]; then
  echo "dir glue_data exist"
  # check checksum
else
  echo "not found dir glue_data, do unzip"
  unzip $FILE4
fi

cd $top_dir_realpath


# do running
export PYTHONPATH="${PYTHONPATH}:${top_dir_realpath}:${bert_dir}"
export DEBUG_MODE=false
export BERT_LARGE=$bert_dir"/pretrained_models/uncased_L-24_H-1024_A-16"
export BERT_BASE=$bert_dir"/pretrained_models/uncased_L-12_H-768_A-12"
cd $top_dir_realpath"/external/TopsModels/common/"
python setup.py install
cd $top_dir_realpath

export ENABLE_INIT_ON_CPU=1
BERT_CKPT_DIR=$BERT_BASE

# run squad ----------------------------------
# rm -rf squad_output
# export SQUAD_DIR=$bert_dir/dataset/squad/v1.1
# export OUT_DIR=$bert_dir/squad_output
# nohup python $bert_dir/run_squad.py \
#   --vocab_file=$BERT_CKPT_DIR/vocab.txt \
#   --bert_config_file=$BERT_CKPT_DIR/bert_config.json \
#   --init_checkpoint=$BERT_CKPT_DIR/bert_model.ckpt \
#   --do_train=True \
#   --do_predict=True \
#   --device=dtu \
#   --train_file=$SQUAD_DIR/train-v1.1.json \
#   --predict_file=$SQUAD_DIR/dev-v1.1.json \
#   --train_batch_size=1 \
#   --learning_rate=5e-6 \
#   --num_train_epochs=0.003 \
#   --max_seq_length=128 \
#   --doc_stride=128 \
#   --output_dir=$OUT_DIR \
#   --use_resource=False \
#   --use_xla=True \
#   --horovod=False \
#   --display_loss_steps=10 \
#   > $top_dir_realpath/log 2>&1 &

runner="gemini_python"

if ! command -v $runner > /dev/null;
then
  echo "using bin/gemini_python.py"
  rm -rf mrpc_output
  export GLUE_DIR=$bert_dir/dataset/glue_data/MRPC
  export OUT_DIR=$bert_dir/mrpc_output
  python $top_dir_realpath/gemini/bin/gemini_python.py $bert_dir/run_classifier.py \
    --do_train=true \
    --do_eval=false \
    --task_name=MRPC \
    --data_dir=${GLUE_DIR}\
    --vocab_file=${BERT_CKPT_DIR}/vocab.txt \
    --bert_config_file=${BERT_CKPT_DIR}/bert_config.json \
    --init_checkpoint=${BERT_CKPT_DIR}/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=1 \
    --learning_rate=2e-5 \
    --num_train_epochs=0.03 \
    --output_dir=${OUT_DIR} \
    > $top_dir_realpath/log 2>&1
else
  echo "using gemini_python"
  rm -rf mrpc_output
  export GLUE_DIR=$bert_dir/dataset/glue_data/MRPC
  export OUT_DIR=$bert_dir/mrpc_output
  gemini_python $bert_dir/run_classifier.py \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=false \
    --data_dir=${GLUE_DIR}\
    --vocab_file=${BERT_CKPT_DIR}/vocab.txt \
    --bert_config_file=${BERT_CKPT_DIR}/bert_config.json \
    --init_checkpoint=${BERT_CKPT_DIR}/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=1 \
    --learning_rate=2e-5 \
    --num_train_epochs=0.03 \
    --output_dir=${OUT_DIR} \
    > $top_dir_realpath/log 2>&1
fi
# run mrpc ---------------------------------------

vim $top_dir_realpath/log
