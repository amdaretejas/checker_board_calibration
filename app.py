import streamlit as st
import cv2
import time

st.set_page_config(page_title="Live Camera Feed", layout="wide")

st.title("📷 Live Camera Feed Dashboard")

# Sidebar settings
st.sidebar.header("Camera Settings")
camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
width = st.sidebar.slider("Width", 320, 1920, 640, step=10)
height = st.sidebar.slider("Height", 240, 1080, 480, step=10)

# Buttons
colA, colB = st.columns([1, 1])
start = colA.button("▶ Start Camera")
stop = colB.button("⛔ Stop Camera")

# Placeholder for live feed
frame_placeholder = st.empty()

# FPS display
fps_text = st.empty()

# Store running state in session
if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False


def start_camera():
    cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    prev_time = 0

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to read from camera")
            break

        # Convert BGR to RGB (Streamlit needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the image
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        fps_text.markdown(f"### ✅ FPS: `{fps:.2f}`")

    cap.release()


if st.session_state.run:
    start_camera()
else:
    st.info("Click ▶ Start Camera to begin live feed.")
