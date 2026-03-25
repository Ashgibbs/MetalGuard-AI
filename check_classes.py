from ultralytics import YOLO

MODEL_PATH = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v2_model_premium\weights\best.pt"
model = YOLO(MODEL_PATH)
print("Model Names:", model.names)
print("Number of Classes:", len(model.names))
