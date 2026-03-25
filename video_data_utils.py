import cv2
import os

# def video_to_frames(video_path, output_dir, extract_every_n_frames=1):
#     """
#     Converts a video file into a series of image frames.
#     This is usually what you want when training a model on new video datasets:
#     you extract frames from the video to use as image training data.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return

#     frame_count = 0
#     saved_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Save only every Nth frame to avoid too many duplicate images
#         if frame_count % extract_every_n_frames == 0:
#             frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
#             cv2.imwrite(frame_filename, frame)
#             saved_count += 1
            
#         frame_count += 1
        
#     cap.release()
#     print(f"Extracted {saved_count} frames from {video_path} into '{output_dir}/'")


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Converts a folder of images into a single video file.
    Use this if you have images and want to simulate a video feed or create a video dataset.
    """
    valid_extensions = (".jpg", ".jpeg", ".png")
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(valid_extensions)]
    images.sort() # Sort alphabetically to maintain order
    
    if not images:
        print(f"No valid images found in {image_folder}")
        return
        
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading first image: {first_image_path}")
        return
        
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video '{output_video_path}' from {len(images)} images...")
    
    for image_name in images:
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        
        # Resize if dimensions don't match the first frame, to prevent cv2 crashes
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
            
        video.write(img)
        
    video.release()
    print(f"Successfully saved video to {output_video_path}")


if __name__ == "__main__":
    print("=== Image to Video Converter ===")
    folder_path = input("Enter the path to your folder of images (e.g. yolo_formatted_data/val/scratches): ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
    else:
        output_name = input("Enter the name for the output video (e.g. output_scratches.mp4): ").strip()
        if not output_name.endswith('.mp4'):
            output_name += '.mp4'
            
        fps_input = input("Enter video FPS (frames per second), default is 10: ").strip()
        fps = int(fps_input) if fps_input.isdigit() else 10
        
        images_to_video(folder_path, output_name, fps=fps)
