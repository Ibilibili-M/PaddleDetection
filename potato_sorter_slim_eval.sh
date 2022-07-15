export CUDA_VISIBLE_DEVICES=0 #windows和Mac下不需要执行该命令
# python tools/eval.py \
# -c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_aughsv_flipud_debug.yml \
# -o weights=solov2_r50_fpn_3x_sorter_lr0005_bs16/model_final

# python tools/eval.py \
# -c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_aughsv_flipud_debug.yml \
# --slim_config configs/slim/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16.yml \
# -o weights=output/solov2_r50_fpn_3x_sorter_lr0005_bs16/model_final

# python tools/eval.py \
# -c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_aughsv_flipud_debug.yml \
# --slim_config configs/slim/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_pact.yml \
# -o weights=output/solov2_r50_fpn_3x_sorter_lr0005_bs16_pact/model_final

python tools/eval.py \
-c configs/solov2/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_960_aughsv_flipud_debug.yml \
--slim_config configs/slim/potato_sorter/solov2_r50_fpn_3x_sorter_lr0005_bs16_prune_pact.yml \
-o weights=output/solov2_r50_fpn_3x_sorter_lr0005_bs16_prune_pact/model_final