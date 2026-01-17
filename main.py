import streamlit as st
import cv2
import time

st.set_page_config(page_title="Live Camera Feed", layout="wide")
st.title("📷 Live Camera Calibration")


# ---------------------------
# Session state defaults
# ---------------------------
st.session_state.setdefault("file_name", "camera1")
st.session_state.setdefault("camera_index", 0)

st.session_state.setdefault("board_x", 6)
st.session_state.setdefault("board_y", 9)
st.session_state.setdefault("block_size", 0.5)

st.session_state.setdefault("filter_type", "normal")
st.session_state.setdefault("calibration_type", "Regular")
st.session_state.setdefault("start_capturing", False)

# IMPORTANT LOOP VARIABLES
st.session_state.setdefault("duration_minutes", 5)
st.session_state.setdefault("box_width", 192)
st.session_state.setdefault("box_height", 144)
st.session_state.setdefault("total_cycles", 2)

# chessboard detect settings
st.session_state.setdefault("detection_wait_ms", 200)

# ✅ TEMP STORAGE FOR CAPTURED IMAGES (whole frames)
st.session_state.setdefault("captured_frames", [])   # stores full frames
st.session_state.setdefault("captured_names", [])    # stores names
st.session_state.setdefault("capture_count", 0)      # counter for names


# ---------------------------
# Sidebar UI
# ---------------------------
st.sidebar.header("Camera Settings")

st.session_state.file_name = st.sidebar.text_input(
    "File Name", st.session_state.file_name, max_chars=30, placeholder="file name"
)

st.session_state.camera_index = st.sidebar.selectbox(
    "Select camera", [0, 1, 2], index=int(st.session_state.camera_index)
)

st.sidebar.subheader("Checker board Dimension")
col1, col2, col3 = st.sidebar.columns([1, 1, 1])

st.session_state.board_x = col1.number_input("Board X", 2, 10, int(st.session_state.board_x))
st.session_state.board_y = col2.number_input("Board Y", 2, 10, int(st.session_state.board_y))
st.session_state.block_size = col3.number_input("Block size", 0.5, 50.0, float(st.session_state.block_size), step=0.5)

st.session_state.filter_type = st.sidebar.selectbox(
    "Select Filter", ["normal", "black&white"],
    index=0 if st.session_state.filter_type == "normal" else 1
)

st.session_state.calibration_type = st.sidebar.selectbox(
    "Calibration Type", ["Regular", "Medium", "High"],
    index=["Regular", "Medium", "High"].index(st.session_state.calibration_type)
)

st.sidebar.subheader("Loop Settings")
st.session_state.duration_minutes = st.sidebar.number_input(
    "Duration (minutes)", 1, 60, int(st.session_state.duration_minutes)
)
st.session_state.box_width = st.sidebar.number_input(
    "Box Width", 10, 640, int(st.session_state.box_width)
)
st.session_state.box_height = st.sidebar.number_input(
    "Box Height", 10, 480, int(st.session_state.box_height)
)

# ✅ Buttons
start_capturing_btn = st.sidebar.button("Start Capturing")
clear_images_btn = st.sidebar.button("Clear Captured Images")  # optional helper

if clear_images_btn:
    st.session_state.captured_frames = []
    st.session_state.captured_names = []
    st.session_state.capture_count = 0


# ---------------------------
# ✅ Sidebar: Captured Images list
# ---------------------------
st.sidebar.subheader("Downloaded Images")

with st.sidebar.container(height=210, border=True):
    if len(st.session_state.captured_names) == 0:
        st.text("No images captured yet.")
    else:
        for name in st.session_state.captured_names:
            st.text(f"- {name}")


# ---------------------------
# Frame size and step calculation
# ---------------------------
frame_size = (640, 480)  # (width, height)

if st.session_state.calibration_type == "Regular":
    calibration_devisior = 4
elif st.session_state.calibration_type == "Medium":
    calibration_devisior = 5
elif st.session_state.calibration_type == "High":
    calibration_devisior = 8
else:
    calibration_devisior = 4

x_add = int(frame_size[0] / calibration_devisior)
y_add = int(frame_size[1] / calibration_devisior)


# ---------------------------
# Display placeholders
# ---------------------------
frame_placeholder = st.empty()
fps_text = st.empty()

st.markdown("### 🔍 Current Box Area (ROI Preview)")
roi_placeholder = st.empty()


def start_camera():
    cap = cv2.VideoCapture(st.session_state.camera_index)

    if not cap.isOpened():
        st.error("❌ Unable to open camera")
        return

    duration_minutes = int(st.session_state.duration_minutes)
    total_cycles = int(st.session_state.total_cycles)

    board_x = int(st.session_state.board_x)
    board_y = int(st.session_state.board_y)

    chessboard_size = (board_x, board_y)

    end_time = time.time() + (duration_minutes * 60)

    box_size = [int(st.session_state.box_width), int(st.session_state.box_height)]

    prev_time = time.time()

    cycle_count = 1
    middle_line = [x_add, y_add]

    while True:
        # timeout exit
        if time.time() >= end_time:
            st.warning("⏳ Stopped because TIME OUT reached.")
            break

        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to read from camera")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # filter
        if st.session_state.filter_type == "black&white":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # rectangle corners from center
        first_corner = [0, 0]
        second_corner = [0, 0]

        first_corner[0] = middle_line[0] - int(box_size[0] / 2)
        first_corner[1] = middle_line[1] - int(box_size[1] / 2)
        second_corner[0] = middle_line[0] + int(box_size[0] / 2)
        second_corner[1] = middle_line[1] + int(box_size[1] / 2)

        # bounds
        if first_corner[0] < 0:
            first_corner[0] = 0
        if first_corner[1] < 0:
            first_corner[1] = 0
        if second_corner[0] > frame_size[0]:
            second_corner[0] = frame_size[0]
        if second_corner[1] > frame_size[1]:
            second_corner[1] = frame_size[1]

        x1, y1 = int(first_corner[0]), int(first_corner[1])
        x2, y2 = int(second_corner[0]), int(second_corner[1])

        # roi crop
        roi = frame[y1:y2, x1:x2]

        found = False
        corners = None

        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(roi_gray, chessboard_size, flags)

            if found:
                roi_draw = roi.copy()
                cv2.drawChessboardCorners(roi_draw, chessboard_size, corners, found)
                roi_placeholder.image(roi_draw, channels="RGB", width=300)
            else:
                roi_placeholder.image(roi, channels="RGB", width=300)

        # draw rectangle: green if found else red
        rect_color = (0, 255, 0) if found else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
        cv2.circle(frame, (int(middle_line[0]), int(middle_line[1])), 6, (0, 0, 255), -1)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        frame_placeholder.image(frame, channels="RGB", use_container_width=True)

        fps_text.markdown(
            f"""
            ### ✅ FPS: `{fps:.2f}`
            **Cycle:** `{cycle_count}/{total_cycles}`
            **Board Size:** `{chessboard_size}`
            **Found:** `{found}`
            **Box Size (W,H):** `{box_size}`
            **Rect:** `{first_corner} -> {second_corner}`
            **Captured:** `{len(st.session_state.captured_frames)}`
            """
        )

        # ✅ MAIN LOGIC
        if found:
            # ✅ STORE FULL FRAME (not ROI)
            st.session_state.capture_count += 1
            img_name = f"capture_{st.session_state.capture_count:03d}"
            print(st.session_state.captured_names)
            st.session_state.captured_names.append(img_name)
            st.session_state.captured_frames.append(frame.copy())  # IMPORTANT: copy

            # move to next location
            middle_line[0] += x_add

            if middle_line[0] >= frame_size[0]:
                middle_line[0] = x_add
                middle_line[1] += y_add

            if middle_line[1] >= frame_size[1]:
                if cycle_count < total_cycles:
                    cycle_count += 1
                    box_size[0], box_size[1] = box_size[1], box_size[0]
                    middle_line = [x_add, y_add]
                    st.success(f"✅ Cycle {cycle_count - 1} completed. Starting Cycle {cycle_count}...")
                else:
                    st.success(f"✅ Cycle {cycle_count} completed. Process finished.")
                    start_capturing_btn = False
                    st.rerun()
                    break
        else:
            time.sleep(st.session_state.detection_wait_ms / 1000)

    cap.release()


# ---------------------------
# Run camera only when Start Capturing is pressed
# ---------------------------
if start_capturing_btn:
    st.session_state.start_capturing = True

if st.session_state.start_capturing:
    start_camera()
else:
    st.info("Press **Start Capturing** to begin the process.")
