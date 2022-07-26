import os
import json
import shutil


input_dir = r"/mnt/disk/lifei/potato_sorter_0715"
output_dir = input_dir + "_single_cls"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count = 0
for root, dirs, files in os.walk(input_dir):
    for f_name in files:
        if f_name.endswith(".json"):
            file_path = os.path.join(root, f_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            for obj_id, obj in enumerate(json_data["shapes"]):
                if obj["label"] != "td":
                    json_data["shapes"][obj_id]["label"] = "df"
            save_path = os.path.join(output_dir, f_name)
            with open(save_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            shutil.copy2(file_path.replace(".json", ".bmp"), output_dir)
            count += 1

print(f"共处理{count}个文件")

