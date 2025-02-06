import cv2
import time
import torch
import numpy as np
import pynvml
import psutil
import csv
from tabulate import tabulate  # For table formatting
from src.face_detection import SCRFD
from src.demographic_analysis import MultiTaskFaceClassifier

# Initialize GPU Monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_usage():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    return {
        "memory_used_MB": mem_info.used / (1024 * 1024),
        "gpu_utilization_%": utilization.gpu
    }

# Load Models
detector = SCRFD(model_file='./weights/scrfd_10g_bnkps.onnx')
detector.prepare(-1)

model_path = './demographic_analysis/checkpoints/best_model.pth'
classifier = MultiTaskFaceClassifier(model_path)

# Open Video File
video_path = './assets/test_vid.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get Video Properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define Video Writer
output_path = './output/processed_video.mp4'  # Output video path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_num = 0
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_num == 1000:
            break
        frame_num +=1

        h, w, _ = frame.shape  # Get frame dimensions

        # Measure Face Detection Inference Time & GPU Usage
        torch.cuda.synchronize()
        start_time = time.time()
        bboxes, kpss = detector.detect(frame, 0.5, input_size=(640, 640))
        torch.cuda.synchronize()
        end_time = time.time()
        face_detection_time = (end_time - start_time) * 1000
        gpu_usage_fd = get_gpu_usage()

        # Skip processing if no faces detected
        num_faces = bboxes.shape[0] if bboxes is not None else 0
        if num_faces == 0:
            out.write(frame)  # Write the original frame if no faces are detected
            continue

        total_ag_time = 0
        total_gpu_util = 0

        for i in range(num_faces):
            x1, y1, x2, y2, score = bboxes[i].astype(int)
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [w, h, w, h])  # Ensure within bounds
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                num_faces -= 1  # Reduce count if face is invalid
                continue

            # Resize face for classifier
            face_resized = cv2.resize(face, (224, 224))

            # Measure Age & Gender Classification Inference Time & GPU Usage
            torch.cuda.synchronize()
            start_time = time.time()
            age_label, gender_label, ethnicity_label = classifier.predict(face_resized)
            torch.cuda.synchronize()
            end_time = time.time()
            age_gender_time = (end_time - start_time) * 1000
            gpu_usage_ag = get_gpu_usage()

            total_ag_time += age_gender_time
            total_gpu_util += gpu_usage_ag['gpu_utilization_%']

            # Draw Bounding Box & Labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{gender_label}, {age_label}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if num_faces > 0:
            avg_ag_time = total_ag_time / num_faces
            avg_gpu_util = total_gpu_util / num_faces
        else:
            avg_ag_time = 0
            avg_gpu_util = 0

        # Print Performance Metrics in Table Format
        table_data = [
            ["Face Detection Time (ms)", f"{face_detection_time:.2f}"],
            ["Face Detection GPU Memory (MB)", f"{gpu_usage_fd['memory_used_MB']:.2f}"],
            ["Face Detection GPU Utilization (%)", f"{gpu_usage_fd['gpu_utilization_%']}%"],
            ["Avg Age/Gender Time (ms)", f"{avg_ag_time:.2f}"],
            ["Avg Age/Gender GPU Utilization (%)", f"{avg_gpu_util:.2f}%"],
            ["CPU Utilization (%)", f"{psutil.cpu_percent():.2f}%"]
        ]

        print("\n" + tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
        
        cv2.imwrite("sample.jpg", frame)
        # Write Processed Frame to Output Video
        out.write(frame)

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pynvml.nvmlShutdown()

print(f"\nProcessed video saved as '{output_path}'")
