import cv2
import random

# Dummy functions for age and gender classification
def classify_age_gender(face_image):
    # Dummy age and gender classification
    gender = random.choice(['Male', 'Female'])
    age = random.choice(['0-18', '19-35', '36-50', '51+'])
    return age, gender

# Initialize the face detector (Haar Cascade or DNN-based detector)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video file
video_path = "video.mp4"  # Path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties (frame width, height, and frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the video
output_path = "output_video.mp4"  # Path where the processed video will be saved
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(faces)
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face for age and gender classification
        face_image = frame[y:y + h, x:x + w]
        age, gender = classify_age_gender(face_image)

        # Display the age and gender
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Write the processed frame to the output video file
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()

# Notify user
print(f"Processed video saved to {output_path}")
