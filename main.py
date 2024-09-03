import time
import cv2
import torch
import streamlit as st
import numpy as np
import pandas as pd
from collections import deque
import threading
import tempfile

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
from utilities.speed_estimator import SpeedEstimator
from ultralytics import YOLO

st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")

# Global variables to control threading
frame_queue = deque(maxlen=1)  
all_speeds = deque(maxlen=500)  
speed_data = []
processing_complete = False

def process_video(model, temp_file_name, selected_ind, conf, iou, enable_trk):
    global processing_complete
    cap = cv2.VideoCapture(temp_file_name)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    line_pts = [(0, h // 2), (w, h // 2)]
    speed_obj = SpeedEstimator(
        reg_pts=line_pts,
        names=model.names,
        view_img=False,  
    )
    
    tracker = 10
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if enable_trk == "Yes":
            tracks = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True, show=False)
        else:
            tracks = model(frame, conf=conf, iou=iou, classes=selected_ind)
        
        frame, speeds = speed_obj.estimate_speed(frame, tracks, return_speed=True)
        all_speeds.extend([speed for _, speed in speeds])
        
        # Convert frame to RGB and add to queue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_queue.append((frame_rgb, time.time() - start_time))
        
        time.sleep(1/fps)

    cap.release()
    processing_complete = True

def inference(model=None):
    """Runs real-time object detection and speed estimation on video input using Ultralytics YOLOv8 in a Streamlit application."""
    check_requirements("streamlit>=1.29.0")

    st.markdown("""<style>MainMenu {visibility: hidden;}</style>""", unsafe_allow_html=True)

    st.title("Speed Estimation App")

    st.sidebar.title("User Configuration")
    source = st.sidebar.selectbox("Video", ("Camera", "Video"))

    vid_file_name = ""
    if source == "Video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
    elif source == "Camera":
        vid_file_name = 0

    # Model selection
    available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolov8")]
    if model:
        available_models.insert(0, model.split(".pt")[0])

    selected_model = st.sidebar.selectbox("Model", available_models)
    with st.spinner("Model is downloading..."):
        # Cuda ON
        model = YOLO(f"{selected_model.lower()}.pt").to("cuda")
        
        # Cuda OFF, Uncomment below and comment above line to run on CPU
        # model = YOLO(f"{selected_model.lower()}.pt")
        class_names = list(model.names.values())
    st.success("Model loaded successfully!")

    # Class selection
    selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
    selected_ind = [class_names.index(option) for option in selected_classes]

    # Inference Config
    enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))
    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

    # Layout
    video_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        chart_placeholder = st.empty()
    with col2:
        metric_placeholder = st.empty()

    if st.sidebar.button("Start"):
        if source == "Video" and vid_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(vid_file.read())
            temp_file.close()
            
            # Thread Video Process
            threading.Thread(target=process_video, args=(model, temp_file.name, selected_ind, conf, iou, enable_trk)).start()
            
            while not processing_complete or frame_queue:
                if frame_queue:
                    frame_rgb, elapsed_time = frame_queue.popleft()
                    video_placeholder.image(frame_rgb, channels="RGB")
                    
                    if len(all_speeds) > 0:
                        avg_speed = np.mean(all_speeds)
                        speed_data.append({'Elapsed Time (s)': elapsed_time, 'Average Speed': avg_speed})
                        
                        df = pd.DataFrame(speed_data)
                        chart_placeholder.line_chart(df.set_index('Elapsed Time (s)'))
                        metric_placeholder.metric("Current Average Speed (km/h)", f"{avg_speed:.2f}")
                
                time.sleep(0.03)  
            
            st.success("Processing complete!")
            
            if speed_data:
                df = pd.DataFrame(speed_data)
                st.text(f'Overall Average Speed: {df["Average Speed"].mean():.2f} km/h')
            else:
                st.text('No speed data recorded')
        
        elif source == "Camera":
            st.warning("Camera input is not implemented in this version.")
        
        else:
            st.warning("Please upload a video file.")

    # Clear CUDA memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    inference()