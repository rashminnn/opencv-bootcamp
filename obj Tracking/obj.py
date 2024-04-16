import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Set up tracker types
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]

# Define input video file name
video_input_file_name = "race_car.mp4"

# Loop through tracker types
for tracker_type in tracker_types:
    # Initialize tracker
    if tracker_type == "BOOSTING":
        tracker = cv2.legacy.TrackerBoosting.create()
    elif tracker_type == "MIL":
        tracker = cv2.legacy.TrackerMIL.create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF.create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT.create()
    elif tracker_type == "TLD":
        tracker = cv2.legacy.TrackerTLD.create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.legacy.TrackerMedianFlow.create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN.create()
    else:
        tracker = cv2.legacy.TrackerMOSSE.create()

    # Read video
    video = cv2.VideoCapture(video_input_file_name)
    ok, frame = video.read()

    # Exit if video not opened
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    else:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define output video file name
    video_output_file_name = f"race_car-{tracker_type}.mp4"
    video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))

    # Define bounding box
    bbox = (1300, 405, 160, 120)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    # Object tracking loop
    while True:
        ok, frame = video.read()

        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Write frame to video
        video_out.write(frame)

    # Release video objects
    video.release()
    video_out.release()

print("Videos saved successfully.")
