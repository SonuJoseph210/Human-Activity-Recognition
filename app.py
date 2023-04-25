import streamlit as st
import vid_pred
import cv2
import shutil
import os



def save_video(video_file):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())
    return "uploaded_video.mp4"

def read_video(path):
    video_file = open(path, 'rb')
    video_bytes = video_file.read()
    return video_bytes

CLASSES_LIST = ['Robbery', 'Explosion', 'Shoplifting', 'Arrest', 'Fighting', 'RoadAccidents']

st.title('Anomaly detection app')
video_file = st.file_uploader("Upload video", type=["mp4"])
if st.button('Predict'):
    if video_file is not None:
        vid_path = save_video(video_file)
        video_bytes = read_video(vid_path)
        st.video(video_bytes)
        pred,prob = vid_pred.vid_class_pred(vid_path,CLASSES_LIST)
        st.markdown(f'Prediction class : {pred}')
        st.markdown(f'Prediction prob : {prob}')
    else:
        st.error('Video file not uploaded')