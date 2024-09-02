import cv2
from ultralytics import YOLO
from utilities import SpeedEstimator
import tempfile
import streamlit as st
import time
from collections import deque
import numpy as np
import pandas as pd

# GPU enabled
model = YOLO("yolov8n.pt").to('cuda')

# CPU only
# model = YOLO("yolov8n.pt")

names = model.model.names

st.title("Speed Estimation Demo")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv2.VideoCapture(temp_file.name)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    line_pts = [(0, h // 2), (w, h // 2)]

    speed_obj = SpeedEstimator(
        reg_pts=line_pts,
        names=names,
        view_img=False,  
    )

    video_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    tracker = 10
    all_speeds = deque(maxlen=int(tracker * fps))  
    speed_data = []
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        tracks = model.track(frame, persist=True, show=False)

        frame, speeds = speed_obj.estimate_speed(frame, tracks, return_speed=True)
        all_speeds.extend([speed for _, speed in speeds])
        
        success, im0 = cap.read()
        if not success:
            break

        # Convert image from BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        video_placeholder.image(frame_rgb, channels="RGB")
        
        current_time = time.time()
        elapsed_time = current_time - start_time
     
        if elapsed_time % tracker < 1 / fps:  
            if all_speeds:
                avg_speed = np.mean(all_speeds)
                speed_data.append({'Elapsed Time (s)': elapsed_time, 'Average Speed': avg_speed})
     
                df = pd.DataFrame(speed_data)
                chart_placeholder.line_chart(df.set_index('Elapsed Time (s)'))

        time.sleep(1/fps)

    cap.release()

    st.success("Processing complete!")
    
    if speed_data:
        df = pd.DataFrame(speed_data)
        st.text(f'Overall Average Speed: {df["Average Speed"].mean():.2f} km/h')
    else:
        st.text('No speed data recorded')
