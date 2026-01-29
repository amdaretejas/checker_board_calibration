import streamlit as st
import numpy as np
import cv2
import queue
import time
import json
import zipfile
import io
import logging

from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Live Camera Feed", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-weight: 800; margin-bottom: 20px;'>
        📷 Live Camera Calibration
    </h1>
    """,
    unsafe_allow_html=True
)

FRAME_SIZE = (640, 480)  # (width, height)

# ✅ RTC Config (IMPORTANT for Cloud)
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# ---------------------------
# Session Defaults
# ---------------------------
st.session_state.setdefault("parameters_page", True)
st.session_state.setdefault("live_capturing_page", False)
st.session_state.setdefault("camera_calibration_page", False)
st.session_state.setdefault("download_page", False)
st.session_state.setdefault("results_page", False)

st.session_state.setdefault("file_name", "camera1")
st.session_state.setdefault("board_x", 6)
st.session_state.setdefault("board_y", 9)
st.session_state.setdefault("block_size", 20)  # in mm
st.session_state.setdefault("filter_type", "normal")
st.session_state.setdefault("calibration_type", "Regular")

st.session_state.setdefault("session_end_time", 300)
st.session_state.setdefault("calibration_devisior", 4)
st.session_state.setdefault("total_cycles", 2)

st.session_state.setdefault("box_width", 192)
st.session_state.setdefault("box_height", 144)

st.session_state.setdefault("captured_frames", [])
st.session_state.setdefault("captured_names", [])
st.session_state.setdefault("capture_count", 0)

st.session_state.setdefault("capture_completed", False)
st.session_state.setdefault("page_transition", False)

# calibration result
st.session_state.setdefault("calibration_zip_bytes", None)
st.session_state.setdefault("calibration_params", None)
st.session_state.setdefault("calibration_done", False)


# ✅ WebRTC
webrtc_ctx = webrtc_streamer(
    key="video-custom",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    desired_playing_state=True,
    video_receiver_size=30,
    sendback_video=False,
)

# ============================================================
# PARAMETERS PAGE
# ============================================================
if st.session_state.parameters_page:
    st.markdown(
        """
        <h3 style='text-align: center; font-weight: 800; margin-bottom: 20px;'>
            Set Parameters
        </h3>
        """,
        unsafe_allow_html=True
    )

    with st.form("calibration_parameters"):

        st.session_state.file_name = st.text_input(
            "File Name", st.session_state.file_name, max_chars=30
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        st.session_state.board_x = col1.number_input("Board X", 3, 20, int(st.session_state.board_x))
        st.session_state.board_y = col2.number_input("Board Y", 4, 20, int(st.session_state.board_y))
        st.session_state.block_size = col3.number_input(
            "Block size (mm)", 5, 100, int(st.session_state.block_size)
        )

        col1, col2 = st.columns([1, 1])
        st.session_state.filter_type = col1.selectbox(
            "Select Filter", ["normal", "black&white"],
            index=0 if st.session_state.filter_type == "normal" else 1
        )
        st.session_state.calibration_type = col2.selectbox(
            "Calibration Type", ["Regular", "Medium", "High"],
            index=["Regular", "Medium", "High"].index(st.session_state.calibration_type)
        )

        start_btn = st.form_submit_button("Start Capturing")

    # divisor set
    if st.session_state.calibration_type == "Regular":
        st.session_state.calibration_devisior = 4
    elif st.session_state.calibration_type == "Medium":
        st.session_state.calibration_devisior = 5
    else:
        st.session_state.calibration_devisior = 8

    if start_btn:
        # reset capture
        st.session_state.captured_frames = []
        st.session_state.captured_names = []
        st.session_state.capture_count = 0

        # reset calibration
        st.session_state.calibration_zip_bytes = None
        st.session_state.calibration_done = False

        # move page
        st.session_state.parameters_page = False
        st.session_state.live_capturing_page = True
        st.session_state.page_transition = True
        st.session_state.capture_completed = False
        st.rerun()


# ============================================================
# LIVE CAPTURE PAGE
# ============================================================
elif st.session_state.live_capturing_page:
    st.markdown(
        """
        <h3 style='text-align: center; font-weight: 800; margin-bottom: 20px;'>
            Live Camera Capture
        </h3>
        """,
        unsafe_allow_html=True
    )

    x_add = int(FRAME_SIZE[0] / st.session_state.calibration_devisior)
    y_add = int(FRAME_SIZE[1] / st.session_state.calibration_devisior)

    col1, col2, col3 = st.columns([1, 2, 1])

    if col1.button("⬅ Back"):
        st.session_state.parameters_page = True
        st.session_state.live_capturing_page = False
        st.rerun()

    status = st.empty()
    image_place = col2.empty()

    if webrtc_ctx.video_receiver is None:
        status.warning("⚠️ Please allow camera permission from browser popup.")
        st.stop()

    # reset once
    if st.session_state.page_transition:
        st.session_state.page_transition = False
        st.session_state.capture_completed = False

    # loop vars
    total_cycles = int(st.session_state.total_cycles)
    chessboard_size = (int(st.session_state.board_x), int(st.session_state.board_y))
    box_size = [int(st.session_state.box_width), int(st.session_state.box_height)]

    start_time = time.time()
    cycle_count = 1
    middle_line = [x_add, y_add]

    while True:

        # timeout
        if time.time() - start_time >= st.session_state.session_end_time:
            status.warning("⏳ TIME OUT reached.")
            break

        if webrtc_ctx.video_receiver is None:
            status.warning("⚠️ Camera disconnected.")
            break

        try:
            video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
        except queue.Empty:
            continue

        frame = video_frame.to_ndarray(format="rgb24")

        if st.session_state.filter_type == "black&white":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # ROI
        x1 = middle_line[0] - int(box_size[0] / 2)
        y1 = middle_line[1] - int(box_size[1] / 2)
        x2 = middle_line[0] + int(box_size[0] / 2)
        y2 = middle_line[1] + int(box_size[1] / 2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(FRAME_SIZE[0], x2), min(FRAME_SIZE[1], y2)

        roi = frame[y1:y2, x1:x2]

        found = False
        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, _ = cv2.findChessboardCorners(roi_gray, chessboard_size, flags)

        rect_color = (0, 255, 0) if found else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)

        image_place.image(frame, channels="RGB", use_container_width=True)

        # found -> capture + move
        if found:
            st.session_state.capture_count += 1
            img_name = f"capture_{st.session_state.capture_count:03d}"
            st.session_state.captured_names.append(img_name)
            st.session_state.captured_frames.append(frame.copy())

            middle_line[0] += x_add

            if middle_line[0] >= FRAME_SIZE[0]:
                middle_line[0] = x_add
                middle_line[1] += y_add

            if middle_line[1] >= FRAME_SIZE[1]:
                if cycle_count < total_cycles:
                    cycle_count += 1
                    box_size[0], box_size[1] = box_size[1], box_size[0]
                    middle_line = [x_add, y_add]
                    status.success(f"✅ Cycle {cycle_count-1} completed. Starting cycle {cycle_count}")
                else:
                    st.session_state.capture_completed = True
                    status.success("✅ Capturing Completed Successfully.")
                    break

        # time.sleep(0.1)

    # move next page if completed
    if st.session_state.capture_completed:
        st.session_state.live_capturing_page = False
        st.session_state.camera_calibration_page = True
        st.rerun()


# ============================================================
# CAPTURED FRAMES PAGE
# ============================================================
elif st.session_state.camera_calibration_page:
    st.markdown(
        """
        <h3 style='text-align: center; font-weight: 800; margin-bottom: 20px;'>
            Captured Frames
        </h3>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    if col1.button("⬅ Back"):
        st.session_state.parameters_page = True
        st.session_state.camera_calibration_page = False
        st.session_state.capture_completed = False
        st.rerun()

    container = col2.container(height=500, border=True)

    if len(st.session_state.captured_names) == 0:
        container.write("No frames captured.")
    else:
        for name in st.session_state.captured_names:
            container.write(f"- {name}")

    if col2.button("Start Calibration"):
        st.session_state.camera_calibration_page = False
        st.session_state.download_page = True
        st.rerun()


# ============================================================
# DOWNLOAD PAGE (Calibration + ZIP)
# ============================================================
elif st.session_state.download_page:

    st.markdown(
        """
        <h3 style='text-align: center; font-weight: 800; margin-bottom: 20px;'>
            Calibration Processing
        </h3>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    if col1.button("⬅ Back"):
        st.session_state.camera_calibration_page = True
        st.session_state.download_page = False
        st.rerun()

    if len(st.session_state.captured_frames) == 0:
        st.warning("No frames to calibrate.")
        st.stop()

    progress = st.progress(0)

    board_x = int(st.session_state.board_x)
    board_y = int(st.session_state.board_y)
    pattern_size = (board_x, board_y)
    block_size = float(st.session_state.block_size)

    # Object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= block_size

    objpoints, imgpoints = [], []

    total = len(st.session_state.captured_frames)

    for idx, frame_rgb in enumerate(st.session_state.captured_frames):
        progress.progress(int(((idx + 1) / total) * 100))

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

    if len(objpoints) < 5:
        st.error("❌ Not enough valid chessboard images for calibration (need at least 5).")
        st.stop()

    image_size = (FRAME_SIZE[0], FRAME_SIZE[1])
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

    # ZIP create
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:

        for frame_rgb, name in zip(st.session_state.captured_frames, st.session_state.captured_names):
            img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", img_bgr)
            if ok:
                zf.writestr(f"Images/{name}.jpg", enc.tobytes())

        zf.writestr("parameters.json", json.dumps(params, indent=4))

    mem_zip.seek(0)

    st.session_state.calibration_zip_bytes = mem_zip.getvalue()
    st.session_state.calibration_done = True

    st.success("✅ Calibration completed successfully!")

    st.download_button(
        label="⬇️ Download Calibration ZIP",
        data=st.session_state.calibration_zip_bytes,
        file_name=f"{st.session_state.file_name}_calibration.zip",
        mime="application/zip",
    )
    if st.button("compair result"):
        st.session_state.download_page = False
        st.session_state.results_page = True
        st.rerun()


# ============================================================
# FINAL RESULTS
# ============================================================
if st.session_state.results_page:

    st.markdown(
        """
        <h3 style='text-align: center; font-weight: 800; margin-bottom: 20px;'>
            Final results
        </h3>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    if col1.button("⬅ Back"):
        st.session_state.download_page = True
        st.session_state.results_page = False
        st.rerun()
    
    status = st.empty()
    col1, col2 = st.columns([1, 1])
    col1.subheader('normal live')
    col2.subheader('modified live')
    normal_image = col1.empty()
    modified_image = col2.empty()
    
    while True:

        if webrtc_ctx.video_receiver is None:
            status.warning("⚠️ Camera disconnected.")
            break

        try:
            video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
        except queue.Empty:
            continue

        frame = video_frame.to_ndarray(format="rgb24")

        if st.session_state.filter_type == "black&white":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        normal_image.image(frame, channels="RGB", use_container_width=True)
        modified_image.image(frame, channels="RGB", use_container_width=True)