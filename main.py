# Importing necessary libraries
from ultralytics import YOLO
import cv2
import time
import os

# Set the model 
model = YOLO('yolov8s_custom.pt')
# Set the source of detection to the default camera
cap = cv2.VideoCapture(0)  # Change '0' to '1' or higher if you have multiple cameras

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create directory if it doesn't exist
os.makedirs("WORKERS/NO_SAFETY", exist_ok=True)

last_check_time = time.time()

while True:
    ret, frame = cap.read()
    # Check if frame was read successfully
    if not ret:
        print("Error: Frame not captured properly. Exiting.")
        break

    # Run YOLO model inference
    results = model(frame, verbose=False)
    classes = []
    safety = ['Glass', 'Gloves', 'Helmet', 'Safety-Vest', 'helmet']

    # Draw bounding boxes for detected safety equipment
    for r in results:
        for c in r.boxes:
            if model.names[int(c.cls)] in safety: 
                class_name = model.names[int(c.cls)]
                classes.append(class_name)
                x1, y1, x2, y2 = map(int, c.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Display the frame
    cv2.imshow('FRAME', frame)

    # Save frames without necessary safety equipment every 15 seconds
    current_time = time.time()
    if current_time - last_check_time >= 15:
        for elm in safety:
            if elm not in classes:
                now = time.localtime()
                filename = f"WORKERS/NO_SAFETY/{now.tm_year}{now.tm_mon:02}{now.tm_mday:02}_{now.tm_hour:02}{now.tm_min:02}{now.tm_sec:02}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                break
        last_check_time = current_time

    # Exit loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
