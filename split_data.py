import splitfolders
import os

# 1. Define your exact local paths
input_dataset_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\Metal-surfaces-defects"
output_formatted_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\yolo_formatted_data"

print("--- STARTING DATA SPLIT ---")
print("Reading images and sorting into train/val folders...")

# 2. Split the data (80% for training, 20% for validation)
splitfolders.ratio(
    input_dataset_dir, 
    output=output_formatted_dir, 
    seed=1337, 
    ratio=(0.8, 0.2), 
    group_prefix=None
)

print(f"\n✅ SUCCESS! Data formatted at: {output_formatted_dir}")
print("You can now safely run the training script.")