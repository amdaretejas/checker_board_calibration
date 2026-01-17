import streamlit as st
import cv2
import time

st.set_page_config(page_title="Live Camera Feed", layout="wide")

st.title("📷 Live Camera Calibration")


st.session_state.setdefault("file_name", 0)
st.session_state.setdefault("camera_index", 0)
st.session_state.setdefault("board_x", 0)
st.session_state.setdefault("board_y", 0)
st.session_state.setdefault("block_size", 0)
st.session_state.setdefault("filter_type", 0)
st.session_state.setdefault("calibration_type", 0)

st.session_state.setdefault("run_button", False)

# Sidebar settings
st.sidebar.header("Camera Settings")
st.session_state.file_name = st.sidebar.text_input("File Name", "camera1", 30, placeholder="file name")
st.session_state.camera_index = st.sidebar.selectbox("select camera", [0,  1 , 2])
st.sidebar.subheader("Checker board Dimantion")
col1, col2, col3 = st.sidebar.columns([1, 1, 1])
st.session_state.board_x = col1.number_input("Board X", 2, 10)
st.session_state.board_y = col2.number_input("Board Y", 2, 10)
st.session_state.block_size = col3.number_input("block size", 0.5, 10.0, 0.5)
st.session_state.filter_type = st.sidebar.selectbox("Select Filter", ["normal", "black&white"])
st.session_state.calibration_type = st.sidebar.selectbox("Calibration Type", ["Regular", "Medium", "High"])
st.sidebar.button("Start Capturing")
st.sidebar.subheader("Downloaded Images")

photoes = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

with st.sidebar.container(height=210, border=True):
    for photo in photoes:
        st.text(f"- {photo}")

frame_size = (640, 480)

if st.session_state.calibration_type == "Regular":
    calibration_devisior = 4
    x_add = int(frame_size[0]/calibration_devisior)
    y_add = int(frame_size[1]/calibration_devisior)
elif st.session_state.calibration_type == "Medium":
    calibration_devisior = 5
    x_add = int(frame_size[0]/calibration_devisior)
    y_add = int(frame_size[1]/calibration_devisior)
elif st.session_state.calibration_type == "High":
    calibration_devisior = 8
    x_add = int(frame_size[0]/calibration_devisior)
    y_add = int(frame_size[1]/calibration_devisior)
else:
    calibration_devisior = 0
    x_add = int(frame_size[0]/calibration_devisior)
    y_add = int(frame_size[1]/calibration_devisior)


st.sidebar.button("Start Calibration")
colA, colB = st.columns([1, 1])
start = colA.button("▶ Start Camera")
stop = colB.button("⛔ Stop Camera")

if start:
    st.session_state.run_button = True
if stop:
    st.session_state.run_button = False

frame_placeholder = st.empty()


# FPS display
fps_text = st.empty()

def start_camera():
    cap = cv2.VideoCapture(st.session_state.camera_index)

    prev_time = 0
    counter = 1

    duration_minutes = 5   # change to 10 if needed
    end_time = time.time() + (duration_minutes * 60)
    box_size = (192, 144)
    middle_line = [x_add, y_add]
    first_corner = [0, 0]
    second_corner = [box_size[0], box_size[1]]
    while time.time() < end_time and st.session_state.run_button:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to read from camera")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if counter%50==0:
            middle_line[0]+=x_add
            first_corner[0] = middle_line[0]-int(box_size[0]/2)
            first_corner[1] = middle_line[1]-int(box_size[1]/2)
            second_corner[0] = middle_line[0]+int(box_size[1]/2)
            second_corner[1] = middle_line[1]+int(box_size[1]/2)
            if first_corner[0] <= 0:
                first_corner[0] = 0
                second_corner[0] = second_corner[0] + abs(first_corner[0])
            if first_corner[1] <= 0:
                first_corner[1] = 0
                second_corner[1] = second_corner[1] + abs(first_corner[1])

            if second_corner[0] >= frame_size[0]:
                second_corner[0] = frame_size[0]
                first_corner[0] = first_corner[0] - abs(first_corner[0])
            if second_corner[1] >= frame_size[1]:
                second_corner[1] = frame_size[1]
                first_corner[1] = first_corner[1] - abs(first_corner[1])

            if middle_line[0] >= frame_size[0]:
                middle_line[1]+=y_add
                middle_line[0] = x_add
                first_corner[0] = middle_line[0]-int(box_size[0]/2)
                first_corner[1] = middle_line[1]-int(box_size[1]/2)
                second_corner[0] = middle_line[0]+int(box_size[1]/2)
                second_corner[1] = middle_line[1]+int(box_size[1]/2)

            if middle_line[1] >= frame_size[1]:
                middle_line = [x_add, y_add]
                first_corner[0] = middle_line[0]-int(box_size[0]/2)
                first_corner[1] = middle_line[1]-int(box_size[1]/2)
                second_corner[0] = middle_line[0]+int(box_size[1]/2)
                second_corner[1] = middle_line[1]+int(box_size[1]/2)

        cv2.rectangle(frame, (first_corner[0], first_corner[1]), (second_corner[0], second_corner[1]), (0, 255, 0), 1)
        cv2.circle(frame, (middle_line[0], middle_line[1]), 5, (255, 0, 0), -1)

        frame_placeholder.image(frame, channels="RGB")

        fps_text.markdown(f"### ✅ FPS: `{counter:.2f}`")
        fps_text.text(f"{first_corner}x{second_corner}")
        counter+=1

    cap.release()

if st.session_state.run_button:
    start_camera()
else:
    st.info("Click ▶ Start Camera to begin live feed.")