import cv2
import numpy as np
from ultralytics import YOLO
from utilities import SpeedEstimator

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("2431853-hd_1920_1080_25fps.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, h//2), (w, h//2)]

# Init speed-estimation obj
speed_obj = SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True,
)

all_speeds = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(frame, persist=True, show=False)

    frame, speeds = speed_obj.estimate_speed(frame, tracks, return_speed=True)
    all_speeds.extend([speed for _, speed in speeds])
    
    video_writer.write(frame)

cap.release()
video_writer.release()

if all_speeds:
       average_speed = np.mean(all_speeds)
       print(f"Average speed: {average_speed:.2f} km/h")
else:
    print("No speed estimations were recorded.")