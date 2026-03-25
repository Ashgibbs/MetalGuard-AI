from ultralytics import YOLO
import sys
import os

def main(version='yolov8s-cls.pt', name="v3_industrial_model"):
    # 1. Point to the dataset (mapped to /content/dataset after unzipping)
    # We use a relative path here because Colab's file structure is different.
    formatted_data_dir = "industrial_defect_dataset"
    project_dir = "runs"

    if not os.path.exists(formatted_data_dir):
        print(f"❌ Error: Dataset directory '{formatted_data_dir}' not found!")
        print("Make sure you unzipped your dataset into the current folder.")
        return

    print(f"--- STARTING TRAINING ON COLAB WITH {version} ---")
    
    # 2. Load the specified model
    model = YOLO(version)

    # 3. Train the model
    results = model.train(
        data=formatted_data_dir,
        epochs=50,             
        imgsz=256,             
        batch=32,               # Colab GPUs can handle larger batches (32+ is good)
        project=project_dir, 
        name=name,
        workers=8,
        patience=10,           
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
    )
    
    print(f"\n✅ TRAINING COMPLETE FOR {version}!")
    print(rf"Weights saved in: {project_dir}/{name}/weights/best.pt")

if __name__ == '__main__':
    model_choice = 'yolov8s-cls.pt'
    run_name = "v3_industrial_model"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'v11':
            model_choice = 'yolo11s-cls.pt'
            run_name = "v11_industrial_model"
        elif sys.argv[1] == 'v26':
            model_choice = 'yolo26s-cls.pt'
            run_name = "v26_industrial_model"
            
    main(model_choice, run_name)
