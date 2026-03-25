
import os

data_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\Metal Surface defect datas\industrial_defect_dataset"
for split in ["train", "val"]:
    split_dir = os.path.join(data_dir, split)
    print(f"\n--- {split} ---")
    if not os.path.exists(split_dir):
        print(f"Directory {split_dir} not found")
        continue
    classes = os.listdir(split_dir)
    for cls in sorted(classes):
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            print(f"{cls}: {count}")
