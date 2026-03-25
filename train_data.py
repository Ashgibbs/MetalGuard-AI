from ultralytics import YOLO
import sys

def main(version='yolov8s-cls.pt', name="v3_industrial_model"):
    # 1. Point to the NEW INDUSTRIAL dataset (PNG format)
    formatted_data_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\Metal Surface defect datas\industrial_defect_dataset"
    project_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project"

    print(f"--- STARTING TRAINING WITH {version} ---")
    
    # 2. Load the specified model (v8, v11, or v26)
    # Note: For classification, suffix '-cls.pt' is usually used for pretrained weights
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
    # Default to YOLOv8 if no args, but can be changed easily
    model_choice = 'yolov8s-cls.pt'
    run_name = "v3_industrial_model"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'v11':
            model_choice = 'yolo11s-cls.pt' # YOLO11 Small Classification
            run_name = "v11_industrial_model"
        elif sys.argv[1] == 'v26':
            model_choice = 'yolo26s-cls.pt' # YOLO26 Small Classification
            run_name = "v26_industrial_model"
            
    main(model_choice, run_name)