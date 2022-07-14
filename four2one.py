"""
将下相机土豆图片数据集转换为分拣位图片数据集
"""
import os
import json
import logging
import argparse
import shutil

import cv2
import random
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def save_json(json_obj, save_path):
    with open(save_path, 'w') as f:
        json.dump(json_obj, f, indent=4)


def group(json_files, group_size=4):
    json_group = []
    image_group = []
    for file_id, file in enumerate(json_files):
        if file_id % group_size == 3:
            json_group.append([json_files[file_id - 3],
                               json_files[file_id - 2],
                               json_files[file_id - 1],
                               json_files[file_id]])
            image_group.append([json_files[file_id - 3].replace('.json', '.bmp'),
                                json_files[file_id - 2].replace('.json', '.bmp'),
                                json_files[file_id - 1].replace('.json', '.bmp'),
                                json_files[file_id].replace('.json', '.bmp')])
    return json_group, image_group


def convert(json_group, image_group, seed, target_size=(1024, 1280), target_dir="./results"):
    for group_id, gp in enumerate(json_group):
        # 裁图&拼图 - orgin shape: (1200, 1920), output shape: (1024, 1280)
        image_list = image_group[group_id]
        target_height, target_width = target_size
        image_joint = np.zeros((target_height * 2, target_width * 2, 3), dtype=np.uint8)
        image_cut_list = []

        json_data = load_json('basic.json')

        image_path_new = os.path.splitext(os.path.basename(gp[0]))[0] + "_joint" + str(seed) + ".bmp"
        json_data["imagePath"] = image_path_new
        json_data["imageHeight"] = 2048
        json_data["imageWidth"] = 2560

        for image_id, image_path in enumerate(image_list):
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
            height, width = image.shape[:2]
            # assert (height == 1200) & (width == 1920), "image shape is not 1200 x 1920"
            # 裁剪
            if height == 1200 or width == 1920:
                gap_y, gap_x = (height - target_height) // 2, (width - target_width) // 2
            else:
                gap_y, gap_x = 0, 0
            image_cut = image[gap_y:height-gap_y, gap_x:width-gap_x, :]
            image_cut_list.append(image_cut)
            # 修改json
            with open(image_path.replace('.bmp', '.json'), 'r') as fi:
                data = json.load(fi)
            for obj in data["shapes"]:
                points = np.array(obj["points"])
                bin_str = {0: "00", 1: "01", 2: "10", 3: "11"}[image_id]
                if image_id != 0:
                    points[:, 0] = points[:, 0] - gap_x + int(bin_str[1]) * target_width
                    points[:, 1] = points[:, 1] - gap_y + int(bin_str[0]) * target_height
                else:
                    points[:, 0] = points[:, 0] - gap_x
                    points[:, 1] = points[:, 1] - gap_y
                json_data["shapes"].append(
                    {
                        "label": obj["label"],
                        "points": points.tolist(),
                        "group_id": obj["group_id"],
                        "shape_type": obj["shape_type"],
                        "flags": {}
                    }
                )

        # 拼图
        image_joint[:target_height, :target_width, :] = image_cut_list[0]
        image_joint[:target_height, target_width:target_width * 2, :] = image_cut_list[1]
        image_joint[target_height:target_height * 2, :target_width, :] = image_cut_list[2]
        image_joint[target_height:target_height * 2, target_width:target_width * 2, :] = image_cut_list[3]

        # 保存图片
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        iamge_save_path = os.path.join(target_dir, image_path_new)
        cv2.imencode('.bmp', image_joint)[1].tofile(iamge_save_path)

        # 保存json
        with open(iamge_save_path.replace(".bmp", ".json"), 'w') as json_f:
            json.dump(json_data, json_f, indent=4)


def get_params():
    parser = argparse.ArgumentParser(description='4合1拼接图片和标签')
    parser.add_argument('--data_dir', type=str, default='', metavar='DATA_DIR', help='图片和标签所在文件夹')
    parser.add_argument('--shuffle', action='store_true', default=False, help='拼接前打乱顺序')
    parser.add_argument('--shuffle_num', type=int, default=1, metavar="SHUFFLE_NUM", help='图片和标签文件打乱次数')
    parser.add_argument('--split', action='store_true', default=False, help='将图片和json文件分开')
    args = parser.parse_args()
    return args


def main(configs):

    logger.info(f"configs: (data_dir, {configs.data_dir}), (shuffle, {configs.shuffle}), "
                f"(shuffle_num, {configs.shuffle_num}), (split, {configs.split})")
    json_files = []
    for root, dirs, files in os.walk(configs.data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    target_dir = os.path.join(os.path.dirname(configs.data_dir),
                              os.path.basename(configs.data_dir) + "_joint")
    logger.info(f"target_dir: {target_dir}")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # 打乱顺序
    for seed in range(configs.shuffle_num):
        if configs.shuffle:
            random.seed(seed)
            random.shuffle(json_files)
        # 图像分组
        json_group, image_group = group(json_files, group_size=4)
        # 读取4张图片，裁剪图片，拼接图片，修改json
        convert(json_group, image_group, seed, target_size=(1024, 1280), target_dir=target_dir)

    # 将图片和json文件放到不同目录
    if configs.split:
        save_dir = target_dir + "_split"
        image_dir = os.path.join(save_dir, 'labelme_imgs')
        json_dir = os.path.join(save_dir, 'labelme_annos')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        for root, dirs, files in os.walk(target_dir):
            for f in files:
                file_path = os.path.join(root, f)
                if f.endswith('.json'):
                    shutil.move(file_path, json_dir)
                elif f.endswith('.bmp'):
                    shutil.move(file_path, image_dir)
                else:
                    raise TypeError(f"不支持的文件格式{os.path.splitext(file)[1]}")
        os.rmdir(target_dir)


if __name__ == '__main__':
    params = get_params()
    # print(**vars(params))
    main(params)
