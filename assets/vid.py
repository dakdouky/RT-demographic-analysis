import cv2
import os

def create_video_from_images(image_dir, output_video_path, image_duration=2, frame_size=(1920, 1080)):
    # Get list of images in the directory, sorted alphabetically (adjust if needed)
    images = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    if not images:
        print("No images found in the directory.")
        return
    
    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    fps = len(images) / (image_duration * len(images))  # To make each image appear for the specified duration
    out = cv2.VideoWriter(output_video_path, fourcc, 1 / image_duration, frame_size)
    
    for image_name in images:
        img_path = os.path.join(image_dir, image_name)
        img = cv2.imread(img_path)
        
        # Resize image to match the frame size (adjust frame_size if needed)
        img_resized = cv2.resize(img, frame_size)
        
        # Write the image to the video file for the specified duration
        for _ in range(int(image_duration * fps)):
            out.write(img_resized)
    
    # Release the VideoWriter object
    out.release()
    print(f"Video created successfully and saved to {output_video_path}")

# Set your image directory and output video path
image_dir = "./"  # Replace with the directory containing your images
output_video_path = "output_video.mp4"  # Replace with your desired output video file path

create_video_from_images(image_dir, output_video_path)
