#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES="3"
export HF_DATASETS_CACHE="/data/renyubing/project_data/ee/cache"

# 根路径
ROOT_DIR="/data/renyubing/project_data"
TASK_DIR=${ROOT_DIR}/ee

# 预训练模型类型
MODEL_TYPE="t5-large"
MODEL_SEM_TYPE="all-mpnet-base-v2"   # sentence transformer 中效果最好模型

# 预训练模型路径
PRE_TRAINED_MODEL_DIR=${ROOT_DIR}/pre_train_model/${MODEL_TYPE}/
PRE_TRAIN_MODEL_DIR=${ROOT_DIR}/pre_train_model/${MODEL_SEM_TYPE}/
FILE_NAME="t5_wiki_setting1_bs8_g3_l_tgt512"
LOG_NAME="t5_wiki_setting1_bs8_g3_l_tgt512_test"

# 微调模型存储路径
filename=$(date +%Y%m%d)_$(date +%H%M%S)
FINETUNE_MODEL_DIR=${TASK_DIR}/RAG_DEE/save_model/t5_baseline
FINETUNE_MODEL_PATH=${FINETUNE_MODEL_DIR}/${FILE_NAME}.pk

LOG_DIR=log
# 创建相关目录
mkdir -p ${FINETUNE_MODEL_DIR}
mkdir -p ${LOG_DIR}

####################用户需提供的数据#####################
# 模型训练、验证、测试文件
FORMAT_DATA_DIR=${TASK_DIR}/RAG_DEE/WIKIEVENTS
TRAIN_DATA_PATH=${FORMAT_DATA_DIR}/prepro_train.jsonl
DEV_DATA_PATH=${FORMAT_DATA_DIR}/prepro_dev.jsonl
TEST_DATA_PATH=${FORMAT_DATA_DIR}/prepro_test.jsonl
TRAIN_DEMO_PATH=${FORMAT_DATA_DIR}/s3train_demo.json
DEV_DEMO_PATH=${FORMAT_DATA_DIR}/s3val_demo.json
TEST_DEMO_PATH=${FORMAT_DATA_DIR}/s3test_demo.json
EVENT_LABEL_PATH=${FORMAT_DATA_DIR}/event_type.txt
ROLE_LABEL_PATH=${FORMAT_DATA_DIR}/role_type.txt

# 日志
LOG_FILE=${LOG_DIR}/${LOG_NAME}.txt
# accelerate launch nohup python\

nohup python run_model/run_t5_baseline.py \
  --do_eval \
  --dataset=wiki \
  --setting_type=contextual_semantic \
  --add_demo=True \
  --add_vec=False \
  --max_demo_len=100 \
  --k_demos=5 \
  --pretrain_model_path=${PRE_TRAINED_MODEL_DIR} \
  --pre_train_model_path=${PRE_TRAIN_MODEL_DIR} \
  --output_dir=${FINETUNE_MODEL_DIR} \
  --model_save_path=${FINETUNE_MODEL_PATH} \
  --train_data_path=${TRAIN_DATA_PATH} \
  --dev_data_path=${DEV_DATA_PATH} \
  --test_data_path=${TEST_DATA_PATH} \
  --train_demo_path=${TRAIN_DEMO_PATH} \
  --dev_demo_path=${DEV_DEMO_PATH} \
  --test_demo_path=${TEST_DEMO_PATH} \
  --event_type_path=${EVENT_LABEL_PATH} \
  --role_type_path=${ROLE_LABEL_PATH} \
  --dataloader_proc_num=4 \
  --epoch_num=40 \
  --per_device_train_batch_size=2 \
  --per_device_eval_batch_size=2 \
  --num_processes=4 \
  --max_input_len=512 \
  --max_target_len=512 \
  --beam_num=1 \
  --ignore_pad_token_for_loss=true \
  --pad_to_max_length=true \
  --learning_rate=5e-5 \
  --weight_decay=0 \
  --warmup_ratio=0 \
  --accumulate_grad_batches=4 \
  --seed=42 \
  >${LOG_FILE} 2>&1 &