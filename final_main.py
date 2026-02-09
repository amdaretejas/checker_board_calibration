import streamlit as st
import numpy as np
import cv2
import queue
import time
import json
import zipfile
import io
import logging
from PIL import Image

from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit_drawable_canvas import st_canvas

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Camera Calibration Pro", layout="wide", page_icon="📷")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        font-weight: 800;
        margin-bottom: 30px;
        color: #1f77b4;
    }
    .sub-header {
        text-align: center;
        font-weight: 600;
        margin-bottom: 20px;
        color: #2c3e50;
    }
    .info-box {
        background-color: #1e1e2f;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #8ab4f8;
        margin: 10px 0;
        color: #e8eaed;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .capture-status {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .countdown {
        font-size: 48px;
        font-weight: bold;
        color: #ff6b6b;
        text-align: center;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 class='main-header'>
        📷 Professional Camera Calibration System
    </h1>
    """,
    unsafe_allow_html=True
)

FRAME_SIZE = (640, 480)  # (width, height)
STABILIZATION_TIME = 1.5  # seconds to wait for image to stabilize
COUNTDOWN_DURATION = 3  # countdown before capture starts

# ✅ RTC Config
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# ---------------------------
# Session State Initialization
# ---------------------------
st.session_state.setdefault("parameters_page", True)
st.session_state.setdefault("live_capturing_page", False)
st.session_state.setdefault("camera_calibration_page", False)
st.session_state.setdefault("download_page", False)
st.session_state.setdefault("results_page", False)

st.session_state.setdefault("file_name", "camera1")
st.session_state.setdefault("board_x", 6)
st.session_state.setdefault("board_y", 9)
st.session_state.setdefault("block_size", 20)
st.session_state.setdefault("filter_type", "normal")
st.session_state.setdefault("calibration_type", "Regular")

st.session_state.setdefault("session_end_time", 1200)
st.session_state.setdefault("calibration_devisior", 4)
st.session_state.setdefault("total_cycles", 2)

st.session_state.setdefault("box_width", 192)
st.session_state.setdefault("box_height", 144)

st.session_state.setdefault("captured_frames", [])
st.session_state.setdefault("captured_names", [])
st.session_state.setdefault("capture_count", 0)

st.session_state.setdefault("capture_completed", False)
st.session_state.setdefault("page_transition", False)

st.session_state.setdefault("calibration_zip_bytes", None)
st.session_state.setdefault("calibration_params", None)
st.session_state.setdefault("calibration_done", False)

st.session_state.setdefault("measure_points", [])
st.session_state.setdefault("mm_per_pixel", None)
st.session_state.setdefault("measurement_frame", None)

# New states for improved UX
st.session_state.setdefault("last_capture_time", 0)
st.session_state.setdefault("stabilization_start", None)
st.session_state.setdefault("countdown_active", False)
st.session_state.setdefault("countdown_start", None)


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
    st.markdown("<h3 class='sub-header'>⚙️ Calibration Parameters</h3>", unsafe_allow_html=True)

    # Instructions
    st.markdown(
        """
        <div class='info-box'>
        <h4>📋 Instructions:</h4>
        <ol>
            <li><b>File Name:</b> Enter a unique name for your calibration session</li>
            <li><b>Chessboard Pattern:</b> Set the internal corners (Board X × Board Y)</li>
            <li><b>Block Size:</b> Enter the size of each square in millimeters</li>
            <li><b>Filter:</b> Choose image processing filter</li>
            <li><b>Calibration Type:</b> Higher quality = more images captured</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("calibration_parameters"):

        st.session_state.file_name = st.text_input(
            "📁 File Name", st.session_state.file_name, max_chars=30,
            help="Unique identifier for this calibration session"
        )

        st.markdown("### 🎯 Chessboard Configuration")
        col1, col2, col3 = st.columns([1, 1, 1])
        st.session_state.board_x = col1.number_input(
            "Board X (columns)", 3, 20, int(st.session_state.board_x),
            help="Number of internal corners horizontally"
        )
        st.session_state.board_y = col2.number_input(
            "Board Y (rows)", 4, 20, int(st.session_state.board_y),
            help="Number of internal corners vertically"
        )
        st.session_state.block_size = col3.number_input(
            "Block size (mm)", 5, 100, int(st.session_state.block_size),
            help="Size of each chessboard square in millimeters"
        )

        st.markdown("### 🔧 Processing Options")
        col1, col2 = st.columns([1, 1])
        st.session_state.filter_type = col1.selectbox(
            "Image Filter", ["normal", "black&white"],
            index=0 if st.session_state.filter_type == "normal" else 1,
            help="Apply filter for better chessboard detection"
        )
        st.session_state.calibration_type = col2.selectbox(
            "Calibration Quality", ["Regular", "Medium", "High"],
            index=["Regular", "Medium", "High"].index(st.session_state.calibration_type),
            help="Higher quality captures more images for better accuracy"
        )

        st.markdown("---")
        start_btn = st.form_submit_button("🚀 Start Calibration Process", use_container_width=True)

    # Quality settings info
    quality_info = {
        "Regular": "~16 images - Quick calibration",
        "Medium": "~25 images - Balanced quality",
        "High": "~64 images - Maximum accuracy"
    }
    
    st.info(f"**Selected Quality:** {quality_info[st.session_state.calibration_type]}")

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
        st.session_state.last_capture_time = 0
        st.session_state.stabilization_start = None

        # reset calibration
        st.session_state.calibration_zip_bytes = None
        st.session_state.calibration_done = False

        # Activate countdown
        st.session_state.countdown_active = True
        st.session_state.countdown_start = time.time()

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
    st.markdown("<h3 class='sub-header'>📸 Live Capture Session</h3>", unsafe_allow_html=True)

    x_add = int(FRAME_SIZE[0] / st.session_state.calibration_devisior)
    y_add = int(FRAME_SIZE[1] / st.session_state.calibration_devisior)

    # Layout
    col_back, col_main, col_info = st.columns([1, 3, 1])

    if col_back.button("⬅ Back to Settings"):
        st.session_state.parameters_page = True
        st.session_state.live_capturing_page = False
        st.session_state.countdown_active = False
        st.rerun()

    # Instructions sidebar
    with col_info:
        st.markdown(
            """
            <div class='info-box'>
            <h4>📌 How it works:</h4>
            <ul>
                <li><b style='color:#28a745'>Green Box:</b> Pattern detected</li>
                <li><b style='color:#dc3545'>Red Box:</b> No pattern</li>
                <li><b>⏱️ Stabilization:</b> 1.5s hold</li>
                <li><b>📸 Auto-capture</b> when ready</li>
                <li><b>⏭️ Skip</b> to next position</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Progress info
        total_expected = (st.session_state.calibration_devisior ** 2) * st.session_state.total_cycles
        st.metric("Target Images", total_expected)
        st.metric("Captured", st.session_state.capture_count)
        
        progress_pct = min((st.session_state.capture_count / total_expected) * 100, 100)
        st.progress(progress_pct / 100)

    status = col_main.empty()
    countdown_display = col_main.empty()
    skip_button_placeholder = col_main.empty()
    image_place = col_main.empty()

    if webrtc_ctx.video_receiver is None:
        status.warning("⚠️ Please allow camera permission from browser popup.")
        st.stop()

    # Countdown before starting
    if st.session_state.countdown_active:
        elapsed = time.time() - st.session_state.countdown_start
        remaining = COUNTDOWN_DURATION - elapsed
        
        if remaining > 0:
            countdown_display.markdown(
                f"""
                <div class='countdown'>
                    Get Ready! {int(remaining) + 1}
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(0.1)
            st.rerun()
        else:
            st.session_state.countdown_active = False
            countdown_display.empty()

    # reset once
    if st.session_state.page_transition:
        st.session_state.page_transition = False
        st.session_state.capture_completed = False
        st.session_state.skip_requested = False

    # loop vars
    total_cycles = int(st.session_state.total_cycles)
    chessboard_size = (int(st.session_state.board_x), int(st.session_state.board_y))
    box_size = [int(st.session_state.box_width), int(st.session_state.box_height)]

    start_time = time.time()
    cycle_count = 1
    middle_line = [x_add, y_add]
    
    # Track last frame to avoid flickering
    last_display_time = 0
    display_interval = 0.1  # Update display every 100ms

    # Skip button - placed outside the loop to avoid flickering
    with skip_button_placeholder:
        if st.button("⏭️ Skip to Next Position", key="skip_btn", use_container_width=True):
            st.session_state.skip_requested = True

    while True:

        # timeout
        if time.time() - start_time >= st.session_state.session_end_time:
            status.warning("⏳ Session timeout reached. Please review captured images.")
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

        # Handle skip request
        if st.session_state.skip_requested:
            st.session_state.skip_requested = False
            st.session_state.stabilization_start = None
            
            # Move to next position
            middle_line[0] += x_add

            if middle_line[0] >= FRAME_SIZE[0]:
                middle_line[0] = x_add
                middle_line[1] += y_add

            if middle_line[1] >= FRAME_SIZE[1]:
                if cycle_count < total_cycles:
                    cycle_count += 1
                    box_size[0], box_size[1] = box_size[1], box_size[0]
                    middle_line = [x_add, y_add]
                    status.info(f"⏭️ Position skipped! Starting cycle {cycle_count}...")
                else:
                    st.session_state.capture_completed = True
                    status.success("🎉 Capture session completed!")
                    break
            
            time.sleep(0.2)
            st.rerun()

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

        # Stabilization logic
        current_time = time.time()
        
        if found:
            # Start stabilization timer
            if st.session_state.stabilization_start is None:
                st.session_state.stabilization_start = current_time
            
            elapsed_stabilization = current_time - st.session_state.stabilization_start
            
            # Visual feedback for stabilization
            if elapsed_stabilization < STABILIZATION_TIME:
                # Show countdown timer
                remaining_time = STABILIZATION_TIME - elapsed_stabilization
                rect_color = (255, 165, 0)  # Orange during stabilization
                
                # Draw timer on frame
                timer_text = f"Hold Steady: {remaining_time:.1f}s"
                cv2.putText(frame, timer_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Draw progress bar
                bar_width = x2 - x1
                progress_width = int(bar_width * (elapsed_stabilization / STABILIZATION_TIME))
                cv2.rectangle(frame, (x1, y2 + 10), (x1 + progress_width, y2 + 20), (255, 165, 0), -1)
                
            else:
                # Ready to capture - green box
                rect_color = (0, 255, 0)
                cv2.putText(frame, "CAPTURING...", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Pattern not found - reset stabilization
            st.session_state.stabilization_start = None
            rect_color = (255, 0, 0)
            cv2.putText(frame, "Position chessboard in box", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 3)

        # Add capture info overlay
        total_expected = (st.session_state.calibration_devisior ** 2) * st.session_state.total_cycles
        info_text = f"Cycle: {cycle_count}/{total_cycles} | Captured: {st.session_state.capture_count}/{total_expected}"
        cv2.putText(frame, info_text, (10, FRAME_SIZE[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Update display only at intervals to reduce flickering
        if current_time - last_display_time >= display_interval:
            image_place.image(frame, channels="RGB", use_container_width=True)
            last_display_time = current_time

        # Capture logic with stabilization
        if found and st.session_state.stabilization_start is not None:
            elapsed_stabilization = current_time - st.session_state.stabilization_start
            
            if elapsed_stabilization >= STABILIZATION_TIME:
                # Image is stable, capture it
                st.session_state.capture_count += 1
                img_name = f"capture_{st.session_state.capture_count:03d}"
                st.session_state.captured_names.append(img_name)
                st.session_state.captured_frames.append(frame.copy())

                # Reset stabilization
                st.session_state.stabilization_start = None
                
                # Move to next position
                middle_line[0] += x_add

                if middle_line[0] >= FRAME_SIZE[0]:
                    middle_line[0] = x_add
                    middle_line[1] += y_add

                if middle_line[1] >= FRAME_SIZE[1]:
                    if cycle_count < total_cycles:
                        cycle_count += 1
                        box_size[0], box_size[1] = box_size[1], box_size[0]
                        middle_line = [x_add, y_add]
                        status.success(f"✅ Cycle {cycle_count-1} completed! Starting cycle {cycle_count}...")
                    else:
                        st.session_state.capture_completed = True
                        status.success("🎉 All images captured successfully!")
                        break

                # Small delay to show capture feedback
                time.sleep(0.2)
                st.rerun()

    # move next page if completed
    if st.session_state.capture_completed:
        st.session_state.live_capturing_page = False
        st.session_state.camera_calibration_page = True
        time.sleep(1)  # Brief pause before transition
        st.rerun()

# ============================================================
# CAPTURED FRAMES PAGE
# ============================================================
elif st.session_state.camera_calibration_page:
    st.markdown("<h3 class='sub-header'>📋 Captured Images Review</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    if col1.button("⬅ Back to Capture"):
        st.session_state.parameters_page = True
        st.session_state.camera_calibration_page = False
        st.session_state.capture_completed = False
        st.rerun()

    with col2:
        st.markdown(
            f"""
            <div class='success-box'>
            <h4>✅ Capture Complete!</h4>
            <p><b>{len(st.session_state.captured_names)}</b> images captured successfully</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        container = st.container(height=400, border=True)

        if len(st.session_state.captured_names) == 0:
            container.write("No frames captured.")
        else:
            # Show in grid format
            cols_per_row = 3
            rows = [st.session_state.captured_names[i:i+cols_per_row] 
                   for i in range(0, len(st.session_state.captured_names), cols_per_row)]
            
            for row in rows:
                container.write(" | ".join([f"✓ {name}" for name in row]))

        st.markdown("---")
        if st.button("🔬 Start Calibration Analysis", use_container_width=True):
            st.session_state.camera_calibration_page = False
            st.session_state.download_page = True
            st.rerun()


# ============================================================
# DOWNLOAD PAGE (Calibration + ZIP)
# ============================================================
elif st.session_state.download_page:

    st.markdown("<h3 class='sub-header'>🔬 Calibration Processing</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    if col1.button("⬅ Back to Review"):
        st.session_state.camera_calibration_page = True
        st.session_state.download_page = False
        st.rerun()

    if len(st.session_state.captured_frames) == 0:
        st.warning("No frames to calibrate.")
        st.stop()

    with col2:
        st.info("🔄 Processing images... Please wait.")
        progress = st.progress(0)
        progress_text = st.empty()

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
        progress_pct = int(((idx + 1) / total) * 100)
        progress.progress(progress_pct / 100)
        progress_text.text(f"Analyzing image {idx + 1} of {total}...")

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

    progress_text.empty()

    if len(objpoints) < 5:
        st.error("❌ Not enough valid chessboard images for calibration (need at least 5).")
        st.stop()

    st.success(f"✅ {len(objpoints)} out of {total} images used for calibration")

    image_size = (FRAME_SIZE[0], FRAME_SIZE[1])
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    st.session_state.calibration_params = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs
    }

    params = {
        "reprojection_error": float(ret),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "used_images": len(objpoints),
        "total_captured": total,
        "pattern_size": pattern_size,
        "block_size": block_size,
    }

    # Display calibration quality
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reprojection Error", f"{ret:.4f} px")
    col_b.metric("Images Used", f"{len(objpoints)}/{total}")
    col_c.metric("Success Rate", f"{(len(objpoints)/total)*100:.1f}%")

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

    # ---- Estimate pixel to mm ratio ----
    sample_img = cv2.cvtColor(st.session_state.captured_frames[0], cv2.COLOR_RGB2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(sample_img, pattern_size)

    if ret_corners:
        corners = corners.reshape(-1, 2)
        dists = np.linalg.norm(corners[1:] - corners[:-1], axis=1)
        avg_square_px = np.mean(dists)
        st.session_state.mm_per_pixel = st.session_state.block_size / avg_square_px
        st.info(f"📏 Calibrated scale: {st.session_state.mm_per_pixel:.4f} mm/pixel")

    st.markdown("---")
    
    col_dl, col_measure = st.columns(2)
    
    col_dl.download_button(
        label="⬇️ Download Calibration Package",
        data=st.session_state.calibration_zip_bytes,
        file_name=f"{st.session_state.file_name}_calibration.zip",
        mime="application/zip",
        use_container_width=True
    )
    
    col_dl, col_measure = st.columns(2)

    npz_buffer = io.BytesIO()
    np.savez(npz_buffer,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs)
    
    col_dl.download_button(
        label="⬇️ Download Calibration File (.npz)",
        data=npz_buffer.getvalue(),
        file_name=f"{st.session_state.file_name}_calibration.npz",
        mime="application/octet-stream",
        use_container_width=True
    )

    if col_measure.button("📏 Measure Distances", use_container_width=True):
        st.session_state.download_page = False
        st.session_state.results_page = True
        st.session_state.measurement_frame = None
        st.rerun()


# ============================================================
# FINAL RESULTS WITH MEASUREMENT
# ============================================================
elif st.session_state.results_page:

    st.markdown("<h3 class='sub-header'>📏 Distance Measurement Tool</h3>", unsafe_allow_html=True)

    # Instructions at top
    st.markdown(
        """
        <div class='info-box'>
        <h4>📌 Measurement Instructions:</h4>
        <ol>
            <li>Click on two points in the <b>undistorted image</b> to measure distance</li>
            <li>Distance will be shown in both <b>pixels</b> and <b>millimeters</b></li>
            <li>Use <b>Capture New Frame</b> to measure on a different image</li>
            <li>Use <b>Clear All Points</b> to reset measurements</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_back, col_spacer = st.columns([1, 5])

    if col_back.button("⬅ Back to Calibration"):
        st.session_state.download_page = True
        st.session_state.results_page = False
        st.rerun()

    if st.session_state.calibration_params is None:
        st.error("Calibration parameters not found. Please calibrate first.")
        st.stop()

    camera_matrix = np.array(st.session_state.calibration_params["camera_matrix"])
    dist_coeffs = np.array(st.session_state.calibration_params["dist_coeffs"])

    # Capture a frame for measurement if not already captured
    if st.session_state.measurement_frame is None:
        if webrtc_ctx.video_receiver is None:
            st.warning("⚠️ Camera disconnected. Please enable camera access.")
            st.stop()

        with st.spinner("📸 Capturing frame for measurement..."):
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=2)
                frame = video_frame.to_ndarray(format="rgb24")

                if st.session_state.filter_type == "black&white":
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                # Undistort the frame
                h, w = frame.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
                undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

                x, y, w_roi, h_roi = roi
                undistorted = undistorted[y:y + h_roi, x:x + w_roi]

                st.session_state.measurement_frame = {
                    "original": frame,
                    "undistorted": undistorted
                }
            except queue.Empty:
                st.warning("⏳ Waiting for camera frame... Please ensure camera is active.")
                st.stop()

    # Display images side by side
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📷 Original (Distorted)")
        st.image(st.session_state.measurement_frame["original"], channels="RGB", use_container_width=True)

    with col_right:
        st.markdown("### 🛠 Undistorted - Click to Measure")
        
        undistorted = st.session_state.measurement_frame["undistorted"]
        
        # Canvas for point selection
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.5)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=Image.fromarray(undistorted),
            update_streamlit=True,
            height=undistorted.shape[0],
            width=undistorted.shape[1],
            drawing_mode="point",
            point_display_radius=6,
            key="measurement_canvas_final",
        )

        # Process canvas data and calculate distance
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            
            if len(objects) > 0:
                # Extract all points
                points = []
                for obj in objects:
                    if obj["type"] == "circle":
                        points.append((obj["left"], obj["top"]))
                
                # Display number of points
                st.info(f"📍 **Points Marked:** {len(points)}")
                
                # If we have at least 2 points, calculate distance
                if len(points) >= 2:
                    # Use the last two points
                    p1 = points[-2]
                    p2 = points[-1]
                    
                    x1, y1 = p1
                    x2, y2 = p2
                    
                    # Calculate pixel distance
                    pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Display results prominently
                    st.markdown(
                        f"""
                        <div class='metric-container'>
                        <h3>📏 Measurement Results</h3>
                        <h2>{pixel_dist:.2f} pixels</h2>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Convert to mm if calibration available
                    if st.session_state.mm_per_pixel is not None:
                        real_dist_mm = pixel_dist * st.session_state.mm_per_pixel
                        st.markdown(
                            f"""
                            <h2 style='color: #ffd700;'>{real_dist_mm:.2f} mm</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display coordinates
                    with st.expander("📍 Detailed Point Information"):
                        st.write(f"**Point 1:** ({x1:.1f}, {y1:.1f})")
                        st.write(f"**Point 2:** ({x2:.1f}, {y2:.1f})")
                        st.write(f"**ΔX:** {abs(x2 - x1):.1f} px")
                        st.write(f"**ΔY:** {abs(y2 - y1):.1f} px")
                        
                        if st.session_state.mm_per_pixel is not None:
                            st.write(f"**Scale Factor:** {st.session_state.mm_per_pixel:.4f} mm/px")

    # Control buttons
    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🔄 Capture New Frame", use_container_width=True):
            st.session_state.measurement_frame = None
            st.rerun()
    
    with btn_col2:
        if st.button("🗑️ Clear All Points", use_container_width=True):
            st.rerun()
    
    with btn_col3:
        if st.button("🏠 New Calibration", use_container_width=True):
            st.session_state.results_page = False
            st.session_state.parameters_page = True
            st.session_state.measurement_frame = None
            st.session_state.captured_frames = []
            st.session_state.captured_names = []
            st.session_state.capture_count = 0
            st.rerun()