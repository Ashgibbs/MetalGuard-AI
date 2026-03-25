import os
import argparse
from ultralytics import YOLO
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Please install required packages: pip install scikit-learn matplotlib seaborn")
    exit(1)

def evaluate_metrics(model_path, val_data_dir):
    print(f"⚙️  Loading model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("✅ Model loaded successfully!")
    print(f"   Model Classes: {list(model.names.values())}\n")
    print(f"🔍 Evaluating on dataset: {val_data_dir}")
    print("   This might take a minute depending on the dataset size...\n")

    y_true = []
    y_pred = []
    
    class_folders = [f for f in os.listdir(val_data_dir) if os.path.isdir(os.path.join(val_data_dir, f))]
    class_folders.sort()

    for class_name in class_folders:
        class_path = os.path.join(val_data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            continue
            
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            
            # Predict
            result = model(img_path, verbose=False)[0]
            
            predicted_class = result.names[result.probs.top1]
            
            y_true.append(class_name)
            y_pred.append(predicted_class)

    if not y_true:
        print("❌ No images found in the validation directory.")
        return

    # Print Classification Report (Precision, Recall, F1)
    print("="*60)
    print("📊 CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
    print("="*60)
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    print("="*60)

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_folders)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_folders, 
                yticklabels=class_folders)
    
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\n📈 Saved confusion matrix plot to: {cm_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Precision and Recall of YOLO Classification Model")
    parser.add_argument("--model", type=str, default=r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v4_model\weights\best.pt", help="Path to best.pt")
    parser.add_argument("--data", type=str, default=r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\yolo_formatted_data\val", help="Path to validation data folder")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found at '{args.model}'. Please specify the exact path using --model flag.")
    elif not os.path.exists(args.data):
        print(f"❌ Validation data not found at '{args.data}'. Please specify the exact path using --data flag.")
    else:
        evaluate_metrics(args.model, args.data)
