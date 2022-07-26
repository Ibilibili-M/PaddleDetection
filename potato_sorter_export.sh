#!/bin/bash
set -e

# 模型导出
python tools/export_model.py \
-c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960.yml \
--output_dir=./inference_model \
-o weights=output/solov2_r50_fpn_3x_sorter_lr0005_bs16_960/best_model


# # 导出YOLOv3模型
# python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
#  -o weights=weights/yolov3_darknet53_270e_coco.pdparams
