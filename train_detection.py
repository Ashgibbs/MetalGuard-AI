from ultralytics import YOLO
import sys
import os

def main(version='yolov8s.pt', name="v1_detection_model"):
    # 1. Point to the NEW OBJECT DETECTION DATASET's data.yaml file
    dataset_yaml_path = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\datasets\Object Detection Dataset\data.yaml"
    project_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project"

    if not os.path.exists(dataset_yaml_path):
        print(f"Error: Dataset yaml file not found at {dataset_yaml_path}")
        return

    print(f"--- STARTING DETECTION TRAINING WITH {version} ---")
    
    # 2. Load the specified detection model (e.g. YOLOv8 or YOLO11)
    # Using small versions by default (no '-cls.pt' suffix)
    model = YOLO(version)

    # 3. Train the model
    # Note: running for 1 epoch as a test run. Change to higher epochs for full training.
    results = model.train(
        data=dataset_yaml_path,
        epochs=1,              # TEST RUN: 1 epoch. Change to 50 for full training.
        imgsz=640,             # Standard image size for detection
        batch=16,
        project=project_dir, 
        name=name,
        workers=0,
        patience=10,           
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
    )
    
    print(f"\n✅ TRAINING COMPLETE FOR {version}!")
    print(rf"Weights saved in: {project_dir}\{name}\weights\best.pt")

if __name__ == '__main__':
    # Default to YOLOv8s if no args
    model_choice = 'yolov8s.pt'
    run_name = "v1_yolo8_detection"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'v11':
            model_choice = 'yolo11s.pt' # YOLO11 Small Detection
            run_name = "v1_yolo11_detection"
        elif sys.argv[1] == 'v8':
            model_choice = 'yolov8s.pt'
            run_name = "v1_yolo8_detection"
            
    main(model_choice, run_name)
