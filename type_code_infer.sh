CONFIG_ROOT="configs/solov2/potato_sorter"
CONFIG_NAME="solov2_r50_fpn_3x_sorter_lr0005_bs16_960_aughsv_flipud"
CONFIG_FILE="${CONFIG_ROOT}/${CONFIG_NAME}.yml"
VDL_LOG_DIR="${CONFIG_ROOT}/vdl_dir/scalar/${CONFIG_NAME}"
DEVICES="1"
TRAIN_OUTPUT="${CONFIG_ROOT}/train"
EXPORT_DIR="${CONFIG_ROOT}/export"
WEIGHTS="${TRAIN_OUTPUT}/${CONFIG_NAME}/best_model"
EVAL_OUTPUT="${CONFIG_ROOT}/eval"


export CUDA_VISIBLE_DEVICES=1 #windows和Mac下不需要执行该命令
python tools/infer.py /home/lifei/PaddleDetection/configs/yolox/yolox_m_300e_type_code.yml \
                    --infer_dir=/home/lifei/PaddleDetection/dataset/type_code/val \
                    --output_dir=infer_output/ \
                    --draw_threshold=0.5 \
                    -o weights=output/yolov3_mobilenet_v1_roadsign/model_final \
                    --use_vdl=True