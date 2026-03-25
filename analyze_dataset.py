import os
from collections import Counter
import yaml

dataset_path = r'C:\Projects\HCLTech\Cosmetic Defect Detection Project\datasets\Object Detection Dataset'
yaml_path = os.path.join(dataset_path, 'data.yaml')

# Read yaml
with open(yaml_path, 'r') as f:
    data_cfg = yaml.safe_load(f)

classes = data_cfg.get('names', [])
print(f"Classes ({len(classes)}): {classes}")

splits = ['train', 'valid', 'test']

for split in splits:
    split_dir = os.path.join(dataset_path, split)
    if not os.path.exists(split_dir):
        print(f"\nSplit '{split}' not found at {split_dir}")
        continue
        
    img_dir = os.path.join(split_dir, 'images')
    lbl_dir = os.path.join(split_dir, 'labels')
    
    num_imgs = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    num_lbls = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
    
    print(f"\n--- {split.upper()} SPLIT ---")
    print(f"Images: {num_imgs}, Labels: {num_lbls}")
    
    if os.path.exists(lbl_dir):
        class_counts = Counter()
        for lbl_file in os.listdir(lbl_dir):
            if lbl_file.endswith('.txt'):
                with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                cls_id = int(parts[0])
                                class_counts[cls_id] += 1
                            except ValueError:
                                pass
        print("Class distribution:")
        for cls_id, count in sorted(class_counts.items()):
            cls_name = classes[cls_id] if cls_id < len(classes) else f"Unknown({cls_id})"
            print(f"  - {cls_name}: {count}")

print("\nAnalysis complete.")
