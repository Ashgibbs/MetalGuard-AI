import cv2
from ultralytics import YOLO
import argparse
import os

def process_video(video_path, model_path, output_path=None):
    """
    Runs YOLO classification inference on a video and saves/displays the output.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # fallback
        
    print(f"Processing video: {width}x{height} at {fps} FPS")

    # Set up video writer if an output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to {output_path}")

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run YOLO inference on the current frame
        # For Classification, YOLO returns the top probabilities
        results = model(frame, verbose=False)
        
        # Extract the highest probability/confidence class
        text = "Scanning..."
        color = (0, 255, 0)
        
        if len(results) > 0:
            if getattr(results[0], 'probs', None) is not None:
                # Classification model
                probs = results[0].probs
                top_class_id = probs.top1
                top_class_name = results[0].names[top_class_id]
                confidence = float(probs.top1conf) * 100
                text = f"{top_class_name}: {confidence:.1f}%"
            elif getattr(results[0], 'boxes', None) is not None and len(results[0].boxes) > 0:
                # Detection model (pick highest confidence box)
                box = results[0].boxes[0]
                top_class_id = int(box.cls[0])
                top_class_name = results[0].names[top_class_id]
                confidence = float(box.conf[0]) * 100
                text = f"Detected {top_class_name}: {confidence:.1f}%"
                
                # Draw boxes for detection
                for b in results[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Determine color (e.g., Green if confidence is high, Red if low, or default)
            color = (0, 255, 0) # Green
            
            # Add a background rectangle for better text visibility
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (10, 10), (10 + text_width, 10 + text_height + baseline + 10), (0, 0, 0), cv2.FILLED)
            
            # Draw the text on the frame
            cv2.putText(frame, text, (10, 10 + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Write the processed frame to the output video
        if out is not None:
            out.write(frame)
            
        # Display the frame (Optional, can be disabled if running in background)
        cv2.imshow("Defect Detection Inference", frame)
        
        # Press 'q' to exit the video window early
        if cv2.waitKey(600) & 0xFF == ord('q'):
            print("Video playback interrupted by user.")
            break

    # Clean up
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Done! Processed {frame_count} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Video Inference")
    parser.add_argument("--model", type=str, 
                      default=r"C:\Projects\HCLTech\Cosmetic Defect Detection Project\defect_project\v7_model\weights\best.pt",
                      help="Path to the trained model weight file (.pt)")
    parser.add_argument("--video", type=str, help="Path to the input video file")
    
    args = parser.parse_args()
    
    print(f"=== YOLO Video Inference Engine ===")
    
    # 1. Handle Model Path
    MODEL_PATH = args.model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model path not found: {MODEL_PATH}")
        # Try relative paths in project_dir
        alt_model = os.path.join("defect_project", "v11_industrial_model", "weights", "best.pt")
        if os.path.exists(alt_model):
            print(f"Found alternative: {alt_model}")
            MODEL_PATH = alt_model

    # 2. Handle Video Input
    video_input = args.video
    if not video_input:
        video_input = input("Enter the path to the video you want to test: ").strip()
    
    video_input = video_input.strip('"').strip("'")
    
    if not os.path.exists(video_input):
        alt_path = os.path.join("Video Datasets", video_input)
        if os.path.exists(alt_path):
            video_input = alt_path
    
    # 3. Process
    base_name = os.path.splitext(os.path.basename(video_input))[0]
    output_video = f"{base_name}_analyzed.mp4"
    
    process_video(video_input, MODEL_PATH, output_video)

