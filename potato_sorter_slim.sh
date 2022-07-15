set -e

export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
-c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_aughsv_flipud_debug.yml \
--slim_config configs/slim/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_prune_pact.yml

# 动转静导出模型
python tools/export_model.py \
-c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_aughsv_flipud_debug.yml \
--slim_config configs/slim/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_prune_pact.yml \
-o weights=output/solov2_r50_fpn_3x_sorter_lr0005_bs16_prune_pact/model_final