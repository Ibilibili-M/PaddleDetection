#!/bin/bash
set -e
DATA_VERSION="data-0715"
DEVICES="3"
# PRETRAIN_WEIGHTS_PATH='configs/solov2/potato_sorter/data-0602/train/solov2_r50_fpn_3x_sorter_lr0005_bs16_1120_aughsv_flipud/best_model'
PRETRAIN_WEIGHTS_PATH=''
CONFIG_NAME="solov2_r50_fpn_3x_sorter_lr0005_bs16_960_one"

CONFIG_ROOT="configs/solov2/top_camera"
CONFIG_FILE="${CONFIG_ROOT}/${CONFIG_NAME}.yml"
VDL_LOG_DIR="${CONFIG_ROOT}/${DATA_VERSION}/vdl_dir/scalar/${CONFIG_NAME}"
TRAIN_OUTPUT="${CONFIG_ROOT}/${DATA_VERSION}/train"
EXPORT_DIR="${CONFIG_ROOT}/${DATA_VERSION}/export"
WEIGHTS="${TRAIN_OUTPUT}/${CONFIG_NAME}/best_model"
EVAL_OUTPUT="${CONFIG_ROOT}/${DATA_VERSION}/eval"

echo ${TRAIN_OUTPUT}
echo ${WEIGHTS}


# 模型训练(GPU单卡训练/GPU多卡训练)
if [ ${#DEVICES} -gt 2 ]
then
    echo "GPU多卡训练"
    export CUDA_VISIBLE_DEVICES=${DEVICES}
    python -m paddle.distributed.launch --gpus ${DEVICES} tools/train.py \
    -c ${CONFIG_FILE} \
    -o save_dir=${TRAIN_OUTPUT} pretrain_weights=${PRETRAIN_WEIGHTS_PATH} \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=${VDL_LOG_DIR}
else
    echo "GPU单卡训练"
    export CUDA_VISIBLE_DEVICES=${DEVICES} #windows和Mac下不需要执行该命令
    python tools/train.py \
    -c ${CONFIG_FILE} \
    -o save_dir=${TRAIN_OUTPUT} pretrain_weights=${PRETRAIN_WEIGHTS_PATH} \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=${VDL_LOG_DIR}
fi

# 模型评估
export CUDA_VISIBLE_DEVICES=${DEVICES:0:1} #windows和Mac下不需要执行该命令
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
