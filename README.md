# 🛡️ MetalGuard AI - Cosmetic Defect Detection

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Ultralytics YOLO](https://img.shields.io/badge/YOLOv8-FF1493?style=for-the-badge&logo=yolo)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)

**MetalGuard AI** is a state-of-the-art computer vision and quality control project designed to automatically detect cosmetic defects (such as cracks, rust, scratches, crazing, and rolled-in scales) on industrial metal surfaces and marble materials. 

By leveraging cutting edge deep-learning models (Ultralytics YOLO) coupled with a modern, high-performance web dashboard, this project completely automates the traditional manual inspection processes on manufacturing lines.

---

## ✨ Key Features

1. **Surface Classification (`v7_model`)**:
   - Upload an image of a metal surface and instantly classify whether it is completely `Normal` or if it contains `Defects`.
2. **Object Detection (`v1_defect detection_model`)**:
   - Accurately localize where the defects are.
   - Draws dynamic bounding boxes around all identified defects with real-time confidence scores.
3. **Video Analysis Pipeline**:
   - Upload continuous feed/video from a manufacturing assembly line.
   - The system analyzes the video frame-by-frame.
   - Generates a compiled `.webm` output video containing visualized bounding boxes.
   - Plays the analyzed video directly in a cinematic, responsive 3-column Dashboard.
4. **Beautiful Glassmorphism UI**:
   - A stunning, highly-responsive React front-end powered by Vite that visualizes metrics natively.

---

## 🏗️ Project Architecture

```text
📦 MetalGuard-AI
 ┣ 📂 defect_project                 # Stores the trained YOLO weights (.pt files)
 ┃ ┣ 📂 v1_defect detection_model    # YOLOv8 Object Detection Weights
 ┃ ┗ 📂 v7_model                     # YOLOv8 Image Classification Weights
 ┣ 📂 Web Dashboard                  # Frontend Codebase
 ┃ ┗ 📂 defect-dashboard             # Vite + React + TS App
 ┃   ┣ 📂 src (App.tsx, index.css)
 ┃   ┗ 📜 package.json
 ┣ 📜 main.py                        # FastAPI Backend Application
 ┣ 📜 train_detection.py             # Model training pipelines (Colab / Local)
 ┣ 📜 evaluate_metrics.py            # Evaluation & Confusion Matrix generation
 ┣ 📜 README.md                      # You are here!
 ┗ 📜 .gitignore                     # Git rules blocking massive dataset uploads
```

---

## 🛠️ Installation & Setup

Before you begin, ensure you have **Python 3.9+** and **Node.js (npm)** installed on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/Ashgibbs/MetalGuard-AI.git
cd MetalGuard-AI
```

### 2. Start the Backend (FastAPI + YOLO)
The backend loads the heavy YOLO models into memory and serves the AI inference endpoints (`/predict`, `/predict-detection`, `/predict-video-detection`).

```bash
# Install required Python packages
pip install fastapi uvicorn ultralytics opencv-python-headless pillow python-multipart

# Start the local development server (runs on Port 8000)
uvicorn main:app --reload
```
*Note: The first time it runs, it may take a few seconds to load the PyTorch weights into memory.*

### 3. Start the Frontend (React Dashboard)
Open a **new terminal window**, and navigate into the dashboard folder:

```bash
cd "Web Dashboard/defect-dashboard"

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

### 4. Open the App
Visit [http://localhost:5173](http://localhost:5173) in your browser. 
Upload an image or a video, and watch the AI seamlessly hunt down cosmetic defects!

---

## 🧠 Model Training

The models in this repository were trained using balanced industrial datasets containing thousands of defect images. The datasets themselves are ignored from this repository to save space, but you can reproduce the training using the provided scripts:

- `train_colab.py`
- `train_detection.py`

Results, F1 curves, and Confusion Matrices generated during training are intentionally preserved in the repository alongside their respective model folders!

---

## 📄 License
This projected is licensed under the MIT License.
