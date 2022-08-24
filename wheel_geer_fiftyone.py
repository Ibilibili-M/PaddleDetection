import fiftyone as fo
from sklearn import datasets

data_name = "wheel_geer"
data_version = ""
dataset_name = data_name + "-" + data_version

dataset_path_prefix = "/home/lifei/PaddleDetection/dataset/" + data_name + "/"
# The directory containing the source images
train_data_path = dataset_path_prefix + "train"
val_data_path = dataset_path_prefix + "val"
# The path to the COCO labels JSON file
train_labels_path = dataset_path_prefix + "annotations/instance_train.json"
# val_labels_path = dataset_path_prefix + "annotations/instance_val.json"
# Import the dataset

if dataset_name in fo.list_datasets():
    """加载数据集"""
    print(f"直接加载数据集")
    dataset = fo.load_dataset(dataset_name)
else:
    print(f"重新创建数据集")
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=train_data_path,
        labels_path=train_labels_path,
        tags="train",
        label_types=["detections", "segmentations"],
        name=dataset_name
    )

    # dataset_val = fo.Dataset.from_dir(
    #     dataset_type=fo.types.COCODetectionDataset,
    #     data_path=val_data_path,
    #     tags="val",
    #     label_types=["detections", "segmentations"],
    #     name=dataset_name+"_val"
    # )

    # # 合并数据集
    # dataset.merge_samples(dataset_val)
    # dataset.persistent = True

# Print the first few samples in the dataset
print(dataset.head())
# Ensures that the App processes are safely launched on Windows
session = fo.launch_app(dataset, port=5372)
session.wait()


