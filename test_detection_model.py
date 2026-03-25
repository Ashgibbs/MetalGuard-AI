from ultralytics import YOLO
import os
import random
import glob

def test_detection_model():
    # 1. Define paths
    model_path = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v1_defect detection_model\v1_yolo8_detection\weights\best.pt"
    
    # Check potential directories for test/val images
    potential_dirs = [
        r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\datasets\Object Detection Dataset\test\images",
        r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\datasets\Object Detection Dataset\valid\images"
    ]
    
    test_data_dir = None
    for d in potential_dirs:
        if os.path.exists(d) and len(os.listdir(d)) > 0:
            test_data_dir = d
            break
            
    if not test_data_dir:
        print("❌ Error: Could not find any test or validation images in the specified directories.")
        return

    output_dir = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\test_results"
    
    print(f"--- TESTING DETECTION MODEL: v1_yolo8_detection ---")
    
    # 2. Load the model
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Classes ({len(model.names)}): {list(model.names.values())}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Select random images
    all_images = glob.glob(os.path.join(test_data_dir, "*.*"))
    # Filter for image extensions
    images = [img for img in all_images if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not images:
        print(f"❌ Error: No images found in {test_data_dir}")
        return
        
    # Pick up to 10 random images
    sample_images = random.sample(images, min(10, len(images)))
    print(f"Running prediction on {len(sample_images)} sample images from '{test_data_dir}'...")
    
    # 4. Run inference and save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Ultralytics model.predict handles saving natively
    results = model.predict(
        source=sample_images,
        save=True,               # Save images with drawn bounding boxes
        project=output_dir,      # Base directory for saved results
        name="det_test",         # Subdirectory name
        exist_ok=True,           # Overwrite existing directory
        conf=0.25                # Confidence threshold
    )
    
    print(f"\n✅ TEST COMPLETE!")
    print(rf"   Results with bounding boxes are saved in: {output_dir}\det_test")

if __name__ == '__main__':
    test_detection_model()
