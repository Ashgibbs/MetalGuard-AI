from ultralytics import YOLO
import os
import random
import numpy as np

def test_v5_model():
    # 1. Define paths
    model_path = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v5_spots_model\weights\best.pt"
    val_data_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\Metal Surface defect datas\industrial_defect_dataset\val"

    print(f"--- TESTING MODEL: v5_spots_model (best.pt) ---")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}. Is training still running?")
        return

    # 2. Load the model
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Classes ({len(model.names)}): {list(model.names.values())}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Run predictions
    print("--- RUNNING PREDICTIONS ON VALIDATION SAMPLES ---\n")
    
    correct = 0
    total = 0
    results_summary = []

    class_folders = [f for f in os.listdir(val_data_dir) 
                     if os.path.isdir(os.path.join(val_data_dir, f))]

    for class_name in sorted(class_folders):
        class_path = os.path.join(val_data_dir, class_name)
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            continue

        sample_images = random.sample(images, min(10, len(images)))
        class_correct = 0

        for img_name in sample_images:
            img_path = os.path.join(class_path, img_name)
            result = model(img_path, verbose=False)
            
            predicted_class = result[0].names[result[0].probs.top1]
            if predicted_class == class_name:
                class_correct += 1
                correct += 1
            total += 1

        accuracy = (class_correct / len(sample_images)) * 100
        status = "✅" if accuracy >= 70 else "⚠️"
        print(f"  {status} [{class_name}] Accuracy: {class_correct}/{len(sample_images)} ({accuracy:.0f}%)")
        results_summary.append((class_name, accuracy))

    overall_accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"✅ TEST COMPLETE!")
    print(f"   Overall Accuracy: {correct}/{total} images ({overall_accuracy:.1f}%)")
    print(f"{'='*50}")

if __name__ == '__main__':
    test_v5_model()
