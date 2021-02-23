#!/bin/bash

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo $script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo $top_dir_realpath
bert_dir=$top_dir_realpath"/external/TopsModels/research/training/tensorflow/nlp/bert"
echo $bert_dir

# download pretrained model, and dataset
mkdir -p $bert_dir"/pretrained_models"
mkdir -p $bert_dir"/dataset"

# download pretrained model
cd $bert_dir"/pretrained_models"

FILE=uncased_L-24_H-1024_A-16.zip
# move to function
if [ -f "$FILE" ]; then
  echo "$FILE exist"
  # check checksum
  # if ok, skip
  # if not ok, goto clean and download
else
  echo "$FILE not exist"
  # clean and download
fi

exit -1
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

cd $top_dir_realpath
# need probe

# set pythonpath
export PYTHONPATH="${PYTHONPATH}:${top_dir_realpath}:${bert_dir}"
export BERT_BASE=$bert_dir/uncased_L-24_H-1024_A-16





# export BERT_BASE=$BASE_DIR/uncased_L-12_H-768_A-12
export CUDA_VISIBLE_DEVICES=6

#gpu
#export CUDA_VISIBLE_DEVICES=1,2
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64

#dtu
##log level
#export LMODULE=MM
#export LMLEVEL=3
#export DTU_OPT_MODE=false
#export SDK_VLOG_LEVEL=1
#export TF_CPP_MIN_VLOG_LEVEL=1

##dump graph
#export XLA_FLAGS="--dtu_enable=memory_pressure_analysis  --xla_dump_hlo_as_text --xla_dump_to=hlo_dump/`date "+%m%d%H%M"`/ --xla_dump_hlo_pass_re='.*'"
#export TF_DUMP_GRAPH_PREFIX=./graph_dump/`date "+%m%d%H%M"`/

# export ENABLE_INIT_ON_CPU=1
#export TF_XLA_FLAGS="--tf_xla_auto_jit=-1 --tf_xla_min_cluster_size=4"
#  export XLA_FLAGS="--dtu_enable=memory_pressure_analysis"
#  export DTU_UMD_FLAGS='ib_pool_size=134217728'
#  export STATIC_MEM_MC_BALANCE=true
#  export REDUCE_HBM_USE_PEAK=true
#export CLUSTER_AS_DEVICE=false

#export SQUAD_DIR=$BASE_DIR/dataset/squad/v1.1
#export OUT_DIR=$BASE_DIR/squad_base
#python run_squad.py \
#  --vocab_file=$BERT_BASE/vocab.txt \
#  --bert_config_file=$BERT_BASE/bert_config.json \
#  --init_checkpoint=$BERT_BASE/bert_model.ckpt \
#  --do_train=True \
#  --do_predict=True \
#  --device=dtu \
#  --train_file=$SQUAD_DIR/train-v1.1.json \
#  --predict_file=$SQUAD_DIR/dev-v1.1.json \
#  --train_batch_size=8\
#  --learning_rate=5e-6 \
#  --num_train_epochs=2.0 \
#  --max_seq_length=384 \
#  --doc_stride=128 \
#  --output_dir=$OUT_DIR \
#  --use_resource=False \
#  --use_xla=False \
#  --horovod=False \
#  --display_loss_steps=100
cd /home/albert/TopsModels/common/
python setup.py install
cd -

rm -rf mrpc_output
export GLUE_DIR=${BASE_DIR}/dataset/glue_data/MRPC
export OUT_DIR=$BASE_DIR/mrpc_output
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=false \
  --data_dir=${GLUE_DIR}\
  --vocab_file=${BERT_BASE}/vocab.txt \
  --bert_config_file=${BERT_BASE}/bert_config.json \
  --init_checkpoint=${BERT_BASE}/bert_model.ckpt \
  --max_seq_length=384 \
  --train_batch_size=11 \
  --learning_rate=2e-5 \
  --num_train_epochs=0.3 \
  --output_dir=${OUT_DIR}


