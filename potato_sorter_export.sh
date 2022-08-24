#!/bin/bash
set -e

# 模型导出
python tools/export_model.py \
-c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_single_cls_pretrain.yml \
--output_dir=./inference_model \
-o weights=configs/solov2/potato_sorter/data-0715-single/train/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_single_cls_pretrain/best_model


# # 导出YOLOv3模型
# python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
#  -o weights=weights/yolov3_darknet53_270e_coco.pdparams
