#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   bash ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e
CONFIG_ROOT="configs/solov2/potato_sorter"
CONFIG_NAME="solov2_r50_enhance_potato_sorter"
CONFIG_FILE="${CONFIG_ROOT}/${CONFIG_NAME}.yml"
VDL_LOG_DIR="${CONFIG_ROOT}/vdl_dir/scalar/${CONFIG_NAME}"
MULTI_GPU="mu"
DEVICES="0,1,2,3"
TRAIN_OUTPUT="${CONFIG_ROOT}/train"
EXPORT_DIR="${CONFIG_ROOT}/export"
WEIGHTS="${TRAIN_OUTPUT}/${CONFIG_NAME}/best_model"
EVAL_OUTPUT="${CONFIG_ROOT}/eval"

# 模型训练(GPU单卡训练/GPU多卡训练)
# if [ ${#DEVICES} -gt 2 ]
# then
#     echo "GPU多卡训练"
#     # export CUDA_VISIBLE_DEVICES=0,1,2,3
#     # python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
#     # -c ${CONFIG_FILE} \
#     # -o save_dir=${SAVE_DIR}
#     # --eval \
#     # --use_vdl=true \
#     # --vdl_log_dir=${VDL_LOG_DIR}
# else
#     echo "GPU单卡训练"
#     # export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
#     # python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml
# fi

# # 模型评估
# export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
# python tools/eval.py \
# -c ${CONFIG_FILE} \
# -o weights=https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_roadsign.pdparams \
# --output_eval
# # 模型导出
# python tools/export_model.py \
# -c ${CONFIG_FILE} \
# --output_dir=./inference_model ^
# -o weights=output/solov2_r50_enhance_potato_sorter_20220422/best_model.pdparams

# python tools/export_model.py \
# -c ${CONFIG_FILE} \
# --output_dir=${EXPORT_DIR} \
# -o weights=${WEIGHTS}