from ultralytics import YOLO
import os
import random

def test_model():
    # 1. Define paths
    model_path = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v4_model\weights\best.pt"
    val_data_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\yolo_formatted_data\val"

    print(f"--- TESTING MODEL: v4_model (best.pt) ---")
    
    # 2. Load the model
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Classes ({len(model.names)}): {list(model.names.values())}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Run predictions on sample images from each class in val folder
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

        # Test up to 5 random images per class
        sample_images = random.sample(images, min(5, len(images)))
        class_correct = 0

        for img_name in sample_images:
            img_path = os.path.join(class_path, img_name)
            result = model(img_path, verbose=False)
            
            predicted_class = result[0].names[result[0].probs.top1]
            confidence = float(result[0].probs.top1conf) * 100
            is_correct = predicted_class == class_name
            if is_correct:
                class_correct += 1
                correct += 1
            total += 1

        accuracy = (class_correct / len(sample_images)) * 100
        status = "✅" if accuracy >= 60 else "⚠️"
        print(f"  {status} [{class_name}] Accuracy: {class_correct}/{len(sample_images)} ({accuracy:.0f}%)")
        results_summary.append((class_name, accuracy))

    # 4. Print overall summary
    overall_accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"✅ TEST COMPLETE!")
    print(f"   Overall Accuracy: {correct}/{total} images ({overall_accuracy:.1f}%)")
    print(f"{'='*50}")

    # 5. Show worst performing classes
    results_summary.sort(key=lambda x: x[1])
    if results_summary:
        print(f"\n📊 Lowest performing classes:")
        for cls, acc in results_summary[:3]:
            print(f"   - {cls}: {acc:.0f}%")

if __name__ == '__main__':
    test_model()
