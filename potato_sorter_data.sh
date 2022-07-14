set -e

input_dir="/mnt/disk/lifei/potato/xxx"
# 图片和标签四合一
python four2one.py \
--data_dir ${input_dir} \
--shuffle \
--shuffle_num 4 \
--split

# labelme格式转coco格式
json_input_dir="${input_dir}_joint_split/labelme_annos"
image_input_dir="${input_dir}_joint_split/labelme_imgs"
output_dir="${input_dir}_coco"

python tools/x2coco.py --dataset_type labelme \
--json_input_dir ${json_input_dir} \
--image_input_dir ${image_input_dir} \
--output_dir ${output_dir} \
--train_proportion 0.9 \
--val_proportion 0.1 \
--test_proportion 0.0

# 软链接到项目目录下
ln -s ${output_dir} "/home/lifei/PaddleDetection/dataset/PotatoSorter"
cd "/home/lifei/PaddleDetection/dataset/PotatoSorter"
ls