import streamlit as st
import numpy as np
import cv2
import time
import json
import zipfile
import io


st.set_page_config(page_title="Live Camera Feed", layout="wide")

st.title("📷 Live Camera Feed Dashboard")

# Sidebar settings
st.sidebar.header("Camera Settings")
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
width = st.sidebar.slider("Width", 320, 1920, 640, step=10)
height = st.sidebar.slider("Height", 240, 1080, 480, step=10)
