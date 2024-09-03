import cv2
from ultralytics import YOLO
from speed_estimator import SpeedEstimator
import tempfile
import streamlit as st
import time
from collections import deque
import numpy as np
import pandas as pd
import threading

# GPU enabled
model = YOLO("yolov8n.pt").to('cuda')

names = model.model.names

st.title("Speed Estimation Demo")
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

# Global variables to control threading
frame_queue = deque(maxlen=1)  # Store only the latest frame
all_speeds = deque(maxlen=500)  # Store speed data for averaging
speed_data = []
processing_complete = False

def process_video(temp_file_name):
    global processing_complete
    cap = cv2.VideoCapture(temp_file_name)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    line_pts = [(0, h // 2), (w, h // 2)]
    speed_obj = SpeedEstimator(
        reg_pts=line_pts,
        names=names,
        view_img=False,  
    )
    
    tracker = 10
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        tracks = model.track(frame, persist=True, show=False)
        frame, speeds = speed_obj.estimate_speed(frame, tracks, return_speed=True)
        all_speeds.extend([speed for _, speed in speeds])
        
        # Convert frame to RGB and add to queue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_queue.append((frame_rgb, time.time() - start_time))
        
        time.sleep(1/fps)

    cap.release()
    processing_complete = True

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    
    # Start video processing in a separate thread
    threading.Thread(target=process_video, args=(temp_file.name,)).start()
    
    # Video placeholder takes full width
    video_placeholder = st.empty()

    # Create two columns for chart and metric below the video
    col1, col2 = st.columns(2)

    with col1:
        chart_placeholder = st.empty()
    
    with col2:
        metric_placeholder = st.empty()

    while not processing_complete or frame_queue:
        if frame_queue:
            frame_rgb, elapsed_time = frame_queue.popleft()
            video_placeholder.image(frame_rgb, channels="RGB")
            
            if len(all_speeds) > 0:
                avg_speed = np.mean(all_speeds)
                speed_data.append({'Elapsed Time (s)': elapsed_time, 'Average Speed': avg_speed})
                
                # Update the chart and speed data
                df = pd.DataFrame(speed_data)
                chart_placeholder.line_chart(df.set_index('Elapsed Time (s)'))
                metric_placeholder.metric("Current Average Speed (km/h)", f"{avg_speed:.2f}")
        
        time.sleep(0.03)  # Control the UI update frequency
    
    st.success("Processing complete!")
    
    if speed_data:
        df = pd.DataFrame(speed_data)
        st.text(f'Overall Average Speed: {df["Average Speed"].mean():.2f} km/h')
    else:
        st.text('No speed data recorded')
