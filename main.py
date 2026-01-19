import streamlit as st
import cv2
import time
import numpy as np
import json
import zipfile
import io

st.set_page_config(page_title="Live Camera Feed", layout="wide")
st.title("📷 Live Camera Calibration")

# ---------------------------
# Session state defaults
# ---------------------------
st.session_state.setdefault("file_name", "camera1")
st.session_state.setdefault("camera_index", 0)

st.session_state.setdefault("board_x", 6)
st.session_state.setdefault("board_y", 9)
st.session_state.setdefault("block_size", 24.0)   # in mm (or any unit)
st.session_state.setdefault("filter_type", "normal")
st.session_state.setdefault("calibration_type", "Regular")
st.session_state.setdefault("start_capturing", False)

# IMPORTANT LOOP VARIABLES
st.session_state.setdefault("duration_minutes", 5)
st.session_state.setdefault("box_width", 192)
st.session_state.setdefault("box_height", 144)
st.session_state.setdefault("total_cycles", 2)
st.session_state.setdefault("detection_wait_ms", 200)

# TEMP STORAGE
st.session_state.setdefault("captured_frames", [])
st.session_state.setdefault("captured_names", [])
st.session_state.setdefault("capture_count", 0)

# UI flow states
st.session_state.setdefault("capture_completed", False)     # True only when all cycles completed
st.session_state.setdefault("capture_failed_timeout", False)
st.session_state.setdefault("selected_capture", None)

# Calibration states
st.session_state.setdefault("start_calibration", False)
st.session_state.setdefault("calibration_done", False)
st.session_state.setdefault("calibration_zip_bytes", None)
st.session_state.setdefault("calibration_params", None)


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

st.session_state.board_x = col1.number_input("Board X", 2, 20, int(st.session_state.board_x))
st.session_state.board_y = col2.number_input("Board Y", 2, 20, int(st.session_state.board_y))
st.session_state.block_size = col3.number_input("Block size", 0.5, 100.0, float(st.session_state.block_size), step=1.0)

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

start_capturing_btn = st.sidebar.button("Start Capturing")
clear_images_btn = st.sidebar.button("Clear Captured Images")

if clear_images_btn:
    st.session_state.captured_frames = []
    st.session_state.captured_names = []
    st.session_state.capture_count = 0
    st.session_state.capture_completed = False
    st.session_state.capture_failed_timeout = False
    st.session_state.start_capturing = False
    st.session_state.start_calibration = False
    st.session_state.calibration_done = False
    st.session_state.calibration_zip_bytes = None
    st.session_state.calibration_params = None


# ---------------------------
# Sidebar: Downloaded Images + Click Preview
# ---------------------------
st.sidebar.subheader("Downloaded Images")

with st.sidebar.container(height=220, border=True):
    if len(st.session_state.captured_names) == 0:
        st.text("No images captured yet.")
    else:
        # ✅ click to preview
        st.session_state.selected_capture = st.selectbox(
            "Select captured image",
            st.session_state.captured_names,
            index=len(st.session_state.captured_names) - 1
        )
        st.text("Captured list:")
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

st.markdown("### 🖼️ Selected Captured Image Preview")
selected_preview_placeholder = st.empty()

st.markdown("### ✅ Status")
status_placeholder = st.empty()


# ---------------------------
# Helper: build zip
# ---------------------------
def build_zip(captured_frames, captured_names, params_dict):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Images/
        for i, (frame_rgb, name) in enumerate(zip(captured_frames, captured_names), start=1):
            # convert RGB -> BGR for imencode then JPEG
            img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", img_bgr)
            if ok:
                zf.writestr(f"Images/{name}.jpg", enc.tobytes())

        # parameters.json
        zf.writestr("parameters.json", json.dumps(params_dict, indent=4))

    mem_zip.seek(0)
    return mem_zip.getvalue()


# ---------------------------
# Calibration function
# ---------------------------
def run_calibration(progress_bar):
    # NOTE: OpenCV expects inner corners count (not squares)
    board_x = int(st.session_state.board_x)
    board_y = int(st.session_state.board_y)

    pattern_size = (board_x, board_y)  # you said boxes input -> if needed subtract 1 later
    block_size = float(st.session_state.block_size)

    # Object points in real-world
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= block_size

    objpoints = []
    imgpoints = []

    total = len(st.session_state.captured_frames)
    if total == 0:
        return None

    for idx, frame_rgb in enumerate(st.session_state.captured_frames):
        progress_bar.progress(int((idx / total) * 100))

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if ret:
            # refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

    progress_bar.progress(100)

    if len(objpoints) < 5:
        # Not enough images for reliable calibration
        return None

    image_size = (frame_size[0], frame_size[1])
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    params = {
        "reprojection_error": float(ret),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "used_images": len(objpoints),
        "total_captured": total,
        "pattern_size": pattern_size,
        "block_size": block_size,
    }
    return params


# ---------------------------
# Main camera loop
# ---------------------------
def start_camera():
    cap = cv2.VideoCapture(st.session_state.camera_index)

    if not cap.isOpened():
        st.error("❌ Unable to open camera")
        return

    # Reset completion flags
    st.session_state.capture_completed = False
    st.session_state.capture_failed_timeout = False

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

    # ✅ prevent multiple captures at same position
    already_captured_here = False

    while True:
        if time.time() >= end_time:
            st.session_state.capture_failed_timeout = True
            st.warning("⏳ Stopped because TIME OUT reached.")
            break

        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to read from camera")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if st.session_state.filter_type == "black&white":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # rectangle corners
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

        rect_color = (0, 255, 0) if found else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
        cv2.circle(frame, (int(middle_line[0]), int(middle_line[1])), 6, (0, 0, 255), -1)

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

        # Show selected capture preview
        if st.session_state.selected_capture in st.session_state.captured_names:
            idx = st.session_state.captured_names.index(st.session_state.selected_capture)
            selected_preview_placeholder.image(
                st.session_state.captured_frames[idx], channels="RGB", use_container_width=True
            )

        # ✅ if chessboard found -> capture ONCE then move
        if found and not already_captured_here:
            already_captured_here = True

            st.session_state.capture_count += 1
            img_name = f"capture_{st.session_state.capture_count:03d}"

            st.session_state.captured_names.append(img_name)
            st.session_state.captured_frames.append(frame.copy())

            # move next
            middle_line[0] += x_add
            already_captured_here = False

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
                    # ✅ COMPLETED SUCCESSFULLY
                    st.session_state.capture_completed = True
                    st.success("✅ Capturing process complete.")
                    st.session_state.start_capturing = False
                    st.rerun()
                    break
        else:
            time.sleep(st.session_state.detection_wait_ms / 1000)

    cap.release()


# ---------------------------
# Start capturing
# ---------------------------
if start_capturing_btn:
    st.session_state.start_capturing = True
    st.session_state.calibration_done = False
    st.session_state.calibration_zip_bytes = None
    st.session_state.calibration_params = None
    st.session_state.capture_completed = False
    st.session_state.capture_failed_timeout = False

if st.session_state.start_capturing:
    status_placeholder.info("📷 Capturing in progress...")
    start_camera()
else:
    if st.session_state.capture_completed:
        status_placeholder.success("✅ Capturing process complete. You can start calibration.")
    elif st.session_state.capture_failed_timeout:
        status_placeholder.warning("⏳ Capturing stopped due to timeout. Calibration not allowed.")
    else:
        status_placeholder.info("Press **Start Capturing** to begin the process.")


# ---------------------------
# ✅ Calibration Button (only if capture completed successfully)
# ---------------------------
if st.session_state.capture_completed:
    if st.button("🛠️ Calibrate Images"):
        st.session_state.start_calibration = True

if st.session_state.start_calibration and st.session_state.capture_completed:
    st.markdown("## ⚙️ Calibration Processing...")
    progress = st.progress(0)

    params = run_calibration(progress)

    if params is None:
        st.error("❌ Calibration failed. Not enough valid checkerboard images.")
        st.session_state.start_calibration = False
    else:
        st.session_state.calibration_params = params

        zip_bytes = build_zip(
            st.session_state.captured_frames,
            st.session_state.captured_names,
            params
        )
        st.session_state.calibration_zip_bytes = zip_bytes
        st.session_state.calibration_done = True
        st.session_state.start_calibration = False
        st.success("✅ Calibration completed successfully!")


# ---------------------------
# ✅ Download ZIP Button
# ---------------------------
if st.session_state.calibration_done and st.session_state.calibration_zip_bytes is not None:
    st.download_button(
        label="⬇️ Download Calibration ZIP",
        data=st.session_state.calibration_zip_bytes,
        file_name=f"{st.session_state.file_name}_calibration.zip",
        mime="application/zip"
    )
