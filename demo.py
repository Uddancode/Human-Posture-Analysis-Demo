import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
import time
import os
from datetime import datetime

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

motion_buffer = deque(maxlen=10)
prev_center = None
prev_time = time.time()

# CSV log file setup
LOG_FILE = "posture_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "posture", "source"]).to_csv(LOG_FILE, index=False)


def log_posture(posture, source):
    """Append posture detection results to CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[timestamp, posture, source]], columns=["timestamp", "posture", "source"])
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)


def classify_posture(keypoints, speed):
    """Classify posture from geometry and motion speed."""
    if keypoints is None or len(keypoints) < 14:
        return "Unknown"

    # Extract key points
    nose_y = keypoints[0][1]
    left_shoulder_y = keypoints[5][1]
    right_shoulder_y = keypoints[6][1]
    left_hip_y = keypoints[11][1]
    right_hip_y = keypoints[12][1]
    left_knee_y = keypoints[13][1]
    right_knee_y = keypoints[14][1]
    left_ankle_y = keypoints[15][1]
    right_ankle_y = keypoints[16][1]

    avg_shoulder = (left_shoulder_y + right_shoulder_y) / 2
    avg_hip = (left_hip_y + right_hip_y) / 2
    avg_knee = (left_knee_y + right_knee_y) / 2
    avg_ankle = (left_ankle_y + right_ankle_y) / 2

    hip_knee_diff = abs(avg_knee - avg_hip)
    shoulder_hip_diff = abs(avg_hip - avg_shoulder)
    shoulder_ankle_diff = abs(avg_ankle - avg_shoulder)

    posture = "Standing"

    if shoulder_ankle_diff < 50:
        posture = "Lying Down"
    elif hip_knee_diff < 50:
        posture = "Sitting"
    elif shoulder_hip_diff < 50:
        posture = "Bending"
    else:
        posture = "Standing"

    # Add motion logic
    if speed > 150:
        posture = "Running"
    elif 50 < speed <= 150:
        posture = "Walking"

    return posture


def process_frame(frame, source_name="Unknown"):
    """Detect keypoints and classify posture on a frame."""
    global prev_center, prev_time
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    posture = "Unknown"

    if len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        center = np.mean(keypoints, axis=0)

        curr_time = time.time()
        speed = 0
        if prev_center is not None:
            dist = np.linalg.norm(center - prev_center)
            dt = curr_time - prev_time
            speed = dist / dt if dt > 0 else 0

        motion_buffer.append(speed)
        avg_speed = np.mean(motion_buffer)
        posture = classify_posture(keypoints, avg_speed)

        cv2.putText(
            annotated,
            posture,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3,
        )

        prev_center = center
        prev_time = curr_time

        # Log posture to CSV
        log_posture(posture, source_name)

    return annotated


def run_video(source, source_name="Video"):
    """Process video file or webcam feed."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    print("\n🎥 Press 'Q' to stop the video/camera feed.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = process_frame(frame, source_name)
        cv2.imshow("Human Posture Analysis", annotated)

        # Press 'q' or 'Q' to exit safely
        if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
            print("✅ Exiting video feed safely...")
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path):
    """Process single image."""
    if not os.path.exists(image_path):
        print("Error: File not found:", image_path)
        return
    frame = cv2.imread(image_path)
    annotated = process_frame(frame, "Image")
    cv2.imshow("Posture Analysis (Image)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("\nSelect Input Type:")
    print("1. Image")
    print("2. Video file")
    print("3. Live webcam")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        path = input("Enter image path: ").strip()
        run_image(path)
    elif choice == "2":
        path = input("Enter video file path: ").strip()
        run_video(path, "Video")
    elif choice == "3":
        print("Starting live webcam...")
        run_video(0, "LiveCam")
    else:
        print("Invalid choice! Please restart and select 1/2/3.")


if __name__ == "__main__":
    main()
