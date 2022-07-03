#!/bin/bash
set -e
CONFIG_ROOT="configs/solov2/potato_sorter"
CONFIG_NAME="solov2_r50_fpn_3x_sorter_lr0001_bs16_aughsv"
CONFIG_FILE="${CONFIG_ROOT}/${CONFIG_NAME}.yml"
VDL_LOG_DIR="${CONFIG_ROOT}/vdl_dir/scalar/${CONFIG_NAME}"
DEVICES="0"
TRAIN_OUTPUT="${CONFIG_ROOT}/train"
EXPORT_DIR="${CONFIG_ROOT}/export"
WEIGHTS="${TRAIN_OUTPUT}/${CONFIG_NAME}/best_model"
EVAL_OUTPUT="${CONFIG_ROOT}/eval"
PRETRAIN_WEIGHTS=

# 模型训练(GPU单卡训练/GPU多卡训练)
if [ ${#DEVICES} -gt 2 ]
then
    echo "GPU多卡训练"
    export CUDA_VISIBLE_DEVICES=${DEVICES}
    python -m paddle.distributed.launch --gpus ${DEVICES} tools/train.py \
    -c ${CONFIG_FILE} \
    -o save_dir=${TRAIN_OUTPUT} \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=${VDL_LOG_DIR}
else
    echo "GPU单卡训练"
    export CUDA_VISIBLE_DEVICES=${DEVICES} #windows和Mac下不需要执行该命令
    python tools/train.py \
    -c ${CONFIG_FILE} \
    -o save_dir=${TRAIN_OUTPUT} \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=${VDL_LOG_DIR}
fi

# 模型评估
export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
python tools/eval.py \
-c ${CONFIG_FILE} \
-o weights="${TRAIN_OUTPUT}/${CONFIG_NAME}/best_model.pdparams" \
--classwise \
--output_eval=${EVAL_OUTPUT}

# 模型导出
python tools/export_model.py \
-c ${CONFIG_FILE} \
--output_dir=${EXPORT_DIR} \
-o weights=${WEIGHTS}
