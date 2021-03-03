#!/bin/bash

script_path=`dirname $0`
script_realpath=`realpath $script_path`
echo "script path = "$script_realpath
top_dir_path=$script_path"/.."
top_dir_realpath=`realpath $top_dir_path`
echo "top dir = "$top_dir_realpath
bert_dir=$top_dir_realpath"/external/TopsModels/research/training/tensorflow/nlp/bert"
echo "bert dir = "$bert_dir


cd $top_dir_realpath

# cp impl
mv $bert_dir"/modeling.py" $bert_dir"/modeling_bak"
mv $bert_dir"/run_classifier.py" $bert_dir"/run_classifier_bak"
cp $top_dir_realpath"/experimental/run_classifier.py" $bert_dir/run_classifier.py
cp $top_dir_realpath"/experimental/modeling.py" $bert_dir/modeling.py


# do running
export PYTHONPATH="${PYTHONPATH}:${top_dir_realpath}:${bert_dir}"
export DEBUG_MODE=false
export BERT_LARGE=$bert_dir"/pretrained_models/uncased_L-24_H-1024_A-16"
export BERT_BASE=$bert_dir"/pretrained_models/uncased_L-12_H-768_A-12"
cd $top_dir_realpath"/external/TopsModels/common/"
# python setup.py install
cd $top_dir_realpath

export ENABLE_INIT_ON_CPU=1
BERT_CKPT_DIR=$BERT_BASE


rm -rf $bert_dir/mrpc_output
export GLUE_DIR=$bert_dir/dataset/glue_data/MRPC
export OUT_DIR=$bert_dir/mrpc_output
python $bert_dir/run_classifier.py \
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

mv $bert_dir"/modeling_bak" $bert_dir"/modeling.py"
mv $bert_dir"/run_classifier_bak" $bert_dir"/run_classifier.py"
vim $top_dir_realpath/log


