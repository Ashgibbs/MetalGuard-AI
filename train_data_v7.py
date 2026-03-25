from ultralytics import YOLO
import sys

def main():
    # 1. Point to the BALANCED INDUSTRIAL dataset
    formatted_data_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\Metal Surface defect datas\industrial_defect_dataset_balanced"
    project_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project"
    
    # We will use YOLO11m-cls (Medium) as the "normal" standard model
    version = 'yolo11m-cls.pt'
    name = "v7_model"

    print(f"--- STARTING TRAINING WITH {version} on BALANCED DATASET ---")
    
    # 2. Load the specified model
    model = YOLO(version)

    # 3. Train the model
    results = model.train(
        data=formatted_data_dir,
        epochs=50,             
        imgsz=256,             
        batch=16,
        project=project_dir, 
        name=name,
        workers=4,
        patience=10,           
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
    )
    
    print(f"\n✅ TRAINING COMPLETE FOR {version}!")
    print(rf"Weights saved in: {project_dir}\{name}\weights\best.pt")

if __name__ == '__main__':
    main()
