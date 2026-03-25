from ultralytics import YOLO
import sys
import os

def fix_yaml_paths(yaml_path):
    """
    Colab often struggles if data.yaml has relative paths (../train/images) 
    that were meant for a different directory structure. 
    This function forces the yaml to use absolute Colab paths.
    """
    import yaml
    
    if not os.path.exists(yaml_path):
        print(f"❌ Error: data.yaml not found at {yaml_path}")
        return False
        
    dataset_dir = os.path.dirname(yaml_path)
    
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        
    # Update paths to be absolute in Colab
    data['train'] = os.path.join(dataset_dir, 'train', 'images').replace('\\', '/')
    data['val'] = os.path.join(dataset_dir, 'valid', 'images').replace('\\', '/')
    data['test'] = os.path.join(dataset_dir, 'test', 'images').replace('\\', '/')
    
    # Save back
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)
        
    print(f"✅ Fixed {yaml_path} to use absolute Colab paths.")
    return True

def main(version='yolov8s.pt', name="v1_yolo_detection"):
    # 1. Point to the data.yaml inside the unzipped folder in Colab
    # We assume the user unzipped the file using: !unzip object_detection_dataset.zip -d dataset/
    dataset_yaml_path = "/content/dataset/data.yaml"
    project_dir = "/content/runs/detect"

    if not fix_yaml_paths(dataset_yaml_path):
        print("\n⚠️ Please make sure you unzipped the dataset into a folder named 'dataset' like this:")
        print("   !unzip -q object_detection_dataset.zip -d dataset/")
        return

    print(f"--- STARTING DETECTION TRAINING ON COLAB WITH {version} ---")
    
    # 2. Load the specified detection model
    model = YOLO(version)

    # 3. Train the model
    results = model.train(
        data=dataset_yaml_path,
        epochs=50,             # 50 epochs is standard for fine-tuning
        imgsz=640,             # 640 is standard for YOLO detection
        batch=16,              # Adjust down if Colab runs out of memory (e.g. 8)
        project=project_dir, 
        name=name,
        workers=8,             # Colab has multiple cores
        patience=15,           
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
    )
    
    print(f"\n✅ TRAINING COMPLETE FOR {version}!")
    print(rf"Weights saved in: {project_dir}/{name}/weights/best.pt")

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
