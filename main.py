from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os
import cv2
import tempfile
import numpy as np
import zipfile
import base64
from typing import List
from pydantic import BaseModel

class FolderPathRequest(BaseModel):
    folder_path: str

# 1. Initialize the FastAPI app
app = FastAPI(title="Cosmetic Defect Detection API")

# Ensure output directory exists for saved videos
os.makedirs("output_videos", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="output_videos"), name="outputs")


# Allow the frontend (React) to communicate with this backend without CORS blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the trained models
MODEL_PATH_CLASS = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v7_model\weights\best.pt"
model_class = YOLO(MODEL_PATH_CLASS)

MODEL_PATH_DETECT = r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v1_defect detection_model\v1_yolo8_detection\weights\best.pt"
model_detect = YOLO(MODEL_PATH_DETECT)


@app.get("/")
def home():
    return {
        "message": "Cosmetic Defect Detection API is running!",
        "endpoints": {
            "/predict": "Upload a single image for defect detection",
            "/predict-multiple": "Upload multiple images for batch defect detection",
            "/predict-folder": "Upload a .zip folder of images for defect detection",
            "/predict-local-folder": "Process all images in a local folder path",
            "/predict-video": "Upload a video file for defect detection summary"
        }
    }

def extract_defect_info(result):
    """Utility to pull top class and confidence from YOLO results (Classif or Detect)"""
    if hasattr(result, 'probs') and result.probs is not None:
        # Classification
        top_class_index = result.probs.top1
        top_class_name = result.names[top_class_index]
        confidence = float(result.probs.top1conf)
        return {
            "defect_type": top_class_name,
            "confidence": round(confidence * 100, 2)
        }
    elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        # Detection
        best_box_idx = result.boxes.conf.argmax()
        top_class_index = int(result.boxes.cls[best_box_idx])
        top_class_name = result.names[top_class_index]
        confidence = float(result.boxes.conf[best_box_idx])
        return {
            "defect_type": top_class_name,
            "confidence": round(confidence * 100, 2)
        }
    return {"defect_type": "None", "confidence": 0.0}

# 3. Predict on a single image
@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run the image through your classification model
        results = model_class(image)
        
        # Use our utility to extract details
        info = extract_defect_info(results[0])
        
        return {
            "status": "success",
            "filename": file.filename,
            "defect_type": info["defect_type"],
            "confidence": info["confidence"]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# 4. Predict on multiple images (Select a whole folder by choosing all files)
@app.post("/predict-multiple")
async def predict_multiple_defects(files: List[UploadFile] = File(...)):
    results_list = []
    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            results = model_class(image)
            
            info = extract_defect_info(results[0])
            
            results_list.append({
                "filename": file.filename,
                "defect_type": info["defect_type"],
                "confidence": info["confidence"],
                "status": "success"
            })
        except Exception as e:
            results_list.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {
        "status": "success",
        "total_files": len(files),
        "results": results_list
    }

# 5. Predict on a zipped folder
@app.post("/predict-folder")
async def predict_folder_defects(file: UploadFile = File(...)):
    """
    Upload a .zip file containing a folder of images.
    The API will extract it and predict defects for all images inside.
    """
    if not file.filename.lower().endswith('.zip'):
        return {"status": "error", "message": "Please upload a .zip file containing images."}
    
    results_list = []
    
    try:
        content = await file.read()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            for filename in z.namelist():
                # Skip directories and non-image files
                if filename.endswith('/') or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    continue
                
                try:
                    with z.open(filename) as f:
                        image_bytes = f.read()
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    results = model_class(image)
                    info = extract_defect_info(results[0])
                    
                    results_list.append({
                        "filename": filename.split('/')[-1], # Just the file name
                        "folder": os.path.dirname(filename),
                        "defect_type": info["defect_type"],
                        "confidence": info["confidence"],
                        "status": "success"
                    })
                except Exception as e:
                    results_list.append({
                        "filename": filename,
                        "status": "error",
                        "message": str(e)
                    })
                    
        return {
            "status": "success",
            "total_files_processed": len(results_list),
            "results": results_list
        }
    except zipfile.BadZipFile:
        return {"status": "error", "message": "Uploaded file is not a valid zip archive."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 6. Predict on a local folder directly (Server must have access to the path)
@app.post("/predict-local-folder")
async def predict_local_folder(request: FolderPathRequest):
    """
    Provide an absolute path to a folder on the server.
    The API will read all images in that directory and predict defects.
    """
    folder_path = request.folder_path
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return {"status": "error", "message": f"Directory not found or invalid: {folder_path}"}
    
    results_list = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    
    # Use os.walk to recursively search all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(valid_extensions) or filename.startswith('.'):
                continue
                
            file_path = os.path.join(root, filename)
            
            try:
                image = Image.open(file_path).convert("RGB")
                results = model_class(image)
                info = extract_defect_info(results[0])
                
                # Keep track of relative path for better context
                rel_path = os.path.relpath(root, folder_path)
                
                results_list.append({
                    "filename": filename,
                    "subfolder": "" if rel_path == "." else rel_path,
                    "defect_type": info["defect_type"],
                    "confidence": info["confidence"],
                    "status": "success"
                })
            except Exception as e:
                results_list.append({
                    "filename": filename,
                    "status": "error",
                    "message": f"Failed to process: {str(e)}"
                })
            
    return {
        "status": "success",
        "total_files_processed": len(results_list),
        "folder_path": folder_path,
        "results": results_list
    }

# 7. Predict on a video file
@app.post("/predict-video")
async def predict_video_defects(file: UploadFile = File(...)):
    try:
        # Save uploaded video to a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"status": "error", "message": f"Could not open video file {file.filename}"}

        detected_defects = {} # Summary of unique defects found
        frame_skip = 5 # Optimize processing speed (every 5th frame)
        frame_count = 0
        processed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                results = model_class(frame, verbose=False)
                if results and len(results) > 0:
                    info = extract_defect_info(results[0])
                    # If any defect is detected that isn't 'None' or empty
                    if info["defect_type"] != "None":
                        defect = info["defect_type"]
                        conf = info["confidence"]
                        # Track best confidence for this defect type
                        if defect not in detected_defects or conf > detected_defects[defect]["max_confidence"]:
                            detected_defects[defect] = {"max_confidence": conf}
                processed_count += 1
            
            frame_count += 1
            
        cap.release()
        
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Convert summary to list for response
        summary = [{"defect_type": k, "max_confidence": v["max_confidence"]} for k, v in detected_defects.items()]

        return {
            "status": "success",
            "video_name": file.filename,
            "total_frames": frame_count,
            "frames_processed": processed_count,
            "detected_defects": summary,
            "overall_status": "Defects Found" if summary else "No Defects Detected"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# 8. Predict single image using Detection Model and return plotted image
@app.post("/predict-detection")
async def predict_defect_detection(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run the image through the detection model
        results = model_detect(image)
        result = results[0]
        
        # Extract drawn image
        plotted_img = result.plot() # BGR numpy array
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', plotted_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract detected boxes info
        detected_defects = []
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i])
                cls_name = result.names[cls_id]
                conf = float(result.boxes.conf[i])
                detected_defects.append({
                    "defect_type": cls_name,
                    "confidence": round(conf * 100, 2)
                })
        
        return {
            "status": "success",
            "filename": file.filename,
            "detected_defects": detected_defects,
            "image_base64": img_base64
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# 9. Predict on a video file using Detection Model
@app.post("/predict-video-detection")
async def predict_video_detection(file: UploadFile = File(...)):
    try:
        # Save uploaded video to a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return {"status": "error", "message": f"Could not open video file {file.filename}"}

        # Video Writer Setup
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if original_fps == 0: original_fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_skip = 5 # Process and write every 5th frame to reduce frame rate
        target_fps = max(1, int(original_fps / frame_skip))
        
        import time
        out_filename = f"vid_{int(time.time())}.webm"
        out_path = os.path.join("output_videos", out_filename)
        fourcc = cv2.VideoWriter_fourcc(*'vp80') # WebM Codec
        out = cv2.VideoWriter(out_path, fourcc, target_fps, (width, height))

        detected_defects = {} # Summary of unique defects found
        frame_count = 0
        processed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                results = model_detect(frame, verbose=False)
                res = results[0]
                annotated_frame = res.plot()
                
                if results and len(results) > 0:
                    info = extract_defect_info(res)
                    # If any defect is detected that isn't 'None' or empty
                    if info["defect_type"] != "None":
                        defect = info["defect_type"]
                        conf = info["confidence"]
                        # Track best confidence for this defect type
                        if defect not in detected_defects or conf > detected_defects[defect]["max_confidence"]:
                            detected_defects[defect] = {"max_confidence": conf}
                processed_count += 1
                
                # Write only the processed frame (lower FPS)
                out.write(annotated_frame)
            
            frame_count += 1
            
        cap.release()
        out.release()
        
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Convert summary to list for response
        summary = [{"defect_type": k, "confidence": v["max_confidence"]} for k, v in detected_defects.items()]
        
        # Calculate full URL for the video assuming default localhost:8000
        # In production this should be the server's domain
        video_url = f"http://localhost:8000/outputs/{out_filename}"

        return {
            "status": "success",
            "video_name": file.filename,
            "total_frames": frame_count,
            "frames_processed": processed_count,
            "detected_defects": summary,
            "video_url": video_url,
            "overall_status": "Defects Found" if summary else "No Defects Detected"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}