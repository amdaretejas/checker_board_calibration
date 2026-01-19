import streamlit as st
import cv2
import time
import numpy as np
import json
import zipfile
import io

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Live Camera Feed", layout="wide")
st.title("📷 Live Camera Calibration (Cloud + User Camera)")


# ---------------------------
# Session state defaults
# ---------------------------
st.session_state.setdefault("file_name", "camera1")

# checkerboard inputs
st.session_state.setdefault("board_x", 6)         # inner corners X
st.session_state.setdefault("board_y", 9)         # inner corners Y
st.session_state.setdefault("block_size", 24.0)   # real square size (mm)

st.session_state.setdefault("filter_type", "normal")
st.session_state.setdefault("calibration_type", "Regular")

# LOOP SETTINGS
st.session_state.setdefault("duration_minutes", 5)
st.session_state.setdefault("box_width", 192)
st.session_state.setdefault("box_height", 144)
st.session_state.setdefault("total_cycles", 2)
st.session_state.setdefault("detection_wait_ms", 150)

# FRAME SETTINGS
FRAME_W, FRAME_H = 640, 480
frame_size = (FRAME_W, FRAME_H)  # (width, height)

# TEMP STORAGE
st.session_state.setdefault("captured_frames", [])
st.session_state.setdefault("captured_names", [])
st.session_state.setdefault("capture_count", 0)

# Capture flow state
st.session_state.setdefault("start_capturing", False)
st.session_state.setdefault("capture_completed", False)
st.session_state.setdefault("capture_failed_timeout", False)
st.session_state.setdefault("capture_start_time", None)

# Selected preview
st.session_state.setdefault("selected_capture", None)

# Latest frame buffer
st.session_state.setdefault("latest_frame", None)

# Calibration states
st.session_state.setdefault("start_calibration", False)
st.session_state.setdefault("calibration_done", False)
st.session_state.setdefault("calibration_zip_bytes", None)
st.session_state.setdefault("calibration_params", None)

# Scan state for moving box
st.session_state.setdefault("scan_state", None)


# ---------------------------
# Sidebar UI
# ---------------------------
st.sidebar.header("Settings")

st.session_state.file_name = st.sidebar.text_input(
    "File Name", st.session_state.file_name, max_chars=30
)

st.sidebar.subheader("Checker board (Inner Corners)")
c1, c2, c3 = st.sidebar.columns([1, 1, 1])
st.session_state.board_x = c1.number_input("Board X", 2, 20, int(st.session_state.board_x))
st.session_state.board_y = c2.number_input("Board Y", 2, 20, int(st.session_state.board_y))
st.session_state.block_size = c3.number_input("Square Size", 1.0, 100.0, float(st.session_state.block_size), step=1.0)

st.session_state.filter_type = st.sidebar.selectbox(
    "Filter", ["normal", "black&white"],
    index=0 if st.session_state.filter_type == "normal" else 1
)

st.session_state.calibration_type = st.sidebar.selectbox(
    "Calibration Type", ["Regular", "Medium", "High"],
    index=["Regular", "Medium", "High"].index(st.session_state.calibration_type)
)

st.sidebar.subheader("Capture Settings")
st.session_state.duration_minutes = st.sidebar.number_input(
    "Duration (minutes)", 1, 60, int(st.session_state.duration_minutes)
)
st.session_state.box_width = st.sidebar.number_input(
    "Box Width", 10, FRAME_W, int(st.session_state.box_width)
)
st.session_state.box_height = st.sidebar.number_input(
    "Box Height", 10, FRAME_H, int(st.session_state.box_height)
)
st.session_state.detection_wait_ms = st.sidebar.number_input(
    "Wait(ms) if not found", 0, 1000, int(st.session_state.detection_wait_ms), step=10
)

# Buttons
start_capturing_btn = st.sidebar.button("Start Capturing")
clear_images_btn = st.sidebar.button("Clear Captured Images")

if clear_images_btn:
    st.session_state.captured_frames = []
    st.session_state.captured_names = []
    st.session_state.capture_count = 0

    st.session_state.start_capturing = False
    st.session_state.capture_completed = False
    st.session_state.capture_failed_timeout = False
    st.session_state.capture_start_time = None

    st.session_state.start_calibration = False
    st.session_state.calibration_done = False
    st.session_state.calibration_zip_bytes = None
    st.session_state.calibration_params = None

    st.session_state.scan_state = None


# Sidebar: Captured images list + click preview
st.sidebar.subheader("Downloaded Images")

with st.sidebar.container(height=240, border=True):
    if len(st.session_state.captured_names) == 0:
        st.text("No images captured yet.")
    else:
        st.session_state.selected_capture = st.selectbox(
            "Select captured image",
            st.session_state.captured_names,
            index=len(st.session_state.captured_names) - 1
        )
        st.text("Captured list:")
        for name in st.session_state.captured_names:
            st.text(f"- {name}")


# ---------------------------
# Step calculation
# ---------------------------
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
main_frame_placeholder = st.empty()
info_placeholder = st.empty()

st.markdown("### 🔍 ROI Preview")
roi_placeholder = st.empty()

st.markdown("### 🖼️ Selected Captured Image Preview")
selected_preview_placeholder = st.empty()

st.markdown("### ✅ Status")
status_placeholder = st.empty()


# ---------------------------
# WebRTC Frame Receiver
# ---------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, frame_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.session_state.latest_frame = img
        return img


webrtc_streamer(
    key="camera",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=False,
)


# ---------------------------
# Helper: ZIP builder
# ---------------------------
def build_zip(captured_frames, captured_names, params_dict):
    mem_zip = io.BytesIO()

    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for frame_rgb, name in zip(captured_frames, captured_names):
            img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", img_bgr)
            if ok:
                zf.writestr(f"Images/{name}.jpg", enc.tobytes())

        zf.writestr("parameters.json", json.dumps(params_dict, indent=4))

    mem_zip.seek(0)
    return mem_zip.getvalue()


# ---------------------------
# Calibration function
# ---------------------------
def run_calibration(progress_bar):
    # Use UI inputs
    board_x = int(st.session_state.board_x)
    board_y = int(st.session_state.board_y)

    pattern_size = (board_x, board_y)  # inner corners
    block_size = float(st.session_state.block_size)

    # object points grid
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= block_size

    objpoints, imgpoints = [], []

    total = len(st.session_state.captured_frames)
    if total == 0:
        return None

    for idx, frame_rgb in enumerate(st.session_state.captured_frames):
        progress_bar.progress(int((idx / total) * 100))

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

    progress_bar.progress(100)

    if len(objpoints) < 5:
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
# Frame processing for capture
# ---------------------------
def process_frame(frame, state):
    board_x = int(st.session_state.board_x)
    board_y = int(st.session_state.board_y)
    chessboard_size = (board_x, board_y)

    box_size = state["box_size"]
    middle_line = state["middle_line"]
    cycle_count = state["cycle_count"]

    # compute corners
    first_corner = [0, 0]
    second_corner = [0, 0]

    first_corner[0] = middle_line[0] - int(box_size[0] / 2)
    first_corner[1] = middle_line[1] - int(box_size[1] / 2)
    second_corner[0] = middle_line[0] + int(box_size[0] / 2)
    second_corner[1] = middle_line[1] + int(box_size[1] / 2)

    # bounds
    if first_corner[0] < 0: first_corner[0] = 0
    if first_corner[1] < 0: first_corner[1] = 0
    if second_corner[0] > frame_size[0]: second_corner[0] = frame_size[0]
    if second_corner[1] > frame_size[1]: second_corner[1] = frame_size[1]

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

    # ✅ capture once per position
    if found and not state["already_captured_here"]:
        state["already_captured_here"] = True

        st.session_state.capture_count += 1
        img_name = f"capture_{st.session_state.capture_count:03d}"

        st.session_state.captured_names.append(img_name)
        st.session_state.captured_frames.append(frame.copy())

        # Move to next location
        middle_line[0] += x_add
        state["already_captured_here"] = False

        if middle_line[0] >= frame_size[0]:
            middle_line[0] = x_add
            middle_line[1] += y_add

        # cycle end
        if middle_line[1] >= frame_size[1]:
            if cycle_count < int(st.session_state.total_cycles):
                state["cycle_count"] += 1
                state["box_size"][0], state["box_size"][1] = state["box_size"][1], state["box_size"][0]
                state["middle_line"] = [x_add, y_add]
            else:
                st.session_state.capture_completed = True
                st.session_state.start_capturing = False

    state["middle_line"] = middle_line
    return frame, state, found, first_corner, second_corner


# ---------------------------
# Start capture button
# ---------------------------
if start_capturing_btn:
    st.session_state.start_capturing = True
    st.session_state.capture_completed = False
    st.session_state.capture_failed_timeout = False
    st.session_state.capture_start_time = time.time()

    st.session_state.start_calibration = False
    st.session_state.calibration_done = False
    st.session_state.calibration_zip_bytes = None
    st.session_state.calibration_params = None

    st.session_state.scan_state = {
        "cycle_count": 1,
        "box_size": [int(st.session_state.box_width), int(st.session_state.box_height)],
        "middle_line": [x_add, y_add],
        "already_captured_here": False,
    }


# ---------------------------
# Main capture loop (rerun-based)
# ---------------------------
if st.session_state.start_capturing:

    status_placeholder.info("📷 Capturing in progress...")

    # timeout check
    if st.session_state.capture_start_time is not None:
        if time.time() - st.session_state.capture_start_time > int(st.session_state.duration_minutes) * 60:
            st.session_state.capture_failed_timeout = True
            st.session_state.start_capturing = False

    if st.session_state.latest_frame is not None:
        frame = st.session_state.latest_frame.copy()

        out_frame, new_state, found, fc, sc = process_frame(frame, st.session_state.scan_state)
        st.session_state.scan_state = new_state

        main_frame_placeholder.image(out_frame, channels="RGB", use_container_width=True)

        info_placeholder.markdown(
            f"""
            **Cycle:** `{new_state["cycle_count"]}/{st.session_state.total_cycles}`  
            **Found:** `{found}`  
            **Captured:** `{len(st.session_state.captured_frames)}`  
            **Rect:** `{fc} -> {sc}`
            """
        )

    time.sleep(st.session_state.detection_wait_ms / 1000)
    st.rerun()

else:
    if st.session_state.capture_completed:
        status_placeholder.success("✅ Capturing process complete.")
    elif st.session_state.capture_failed_timeout:
        status_placeholder.warning("⏳ Capturing stopped due to timeout. Calibration not allowed.")
    else:
        status_placeholder.info("Press **Start Capturing** to begin the process.")


# ---------------------------
# Selected capture preview
# ---------------------------
if st.session_state.selected_capture in st.session_state.captured_names:
    idx = st.session_state.captured_names.index(st.session_state.selected_capture)
    selected_preview_placeholder.image(
        st.session_state.captured_frames[idx],
        channels="RGB",
        use_container_width=True
    )


# ---------------------------
# ✅ Calibration button (only if fully completed)
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
# ✅ Download ZIP button
# ---------------------------
if st.session_state.calibration_done and st.session_state.calibration_zip_bytes is not None:
    st.download_button(
        label="⬇️ Download Calibration ZIP",
        data=st.session_state.calibration_zip_bytes,
        file_name=f"{st.session_state.file_name}_calibration.zip",
        mime="application/zip"
    )
