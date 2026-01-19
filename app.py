import streamlit as st
import numpy as np
import cv2
import json
import zipfile
import io
import logging
import queue
import time

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

# ---------------------------
# Pages (permission page removed)
# ---------------------------
st.session_state.setdefault("parameters_page", True)
st.session_state.setdefault("live_capturing_page", False)
st.session_state.setdefault("camera_calibration_page", False)
st.session_state.setdefault("download_page", False)

# ---------------------------
# Buttons / states
# ---------------------------
st.session_state.setdefault("back", False)
st.session_state.setdefault("start_capturing", False)
st.session_state.setdefault("start_calibrating", False)
st.session_state.setdefault("download_zip", False)
st.session_state.setdefault("page_transition", False)

# ---------------------------
# variables
# ---------------------------
st.session_state.setdefault("file_name", "camera1")
st.session_state.setdefault("board_x", 6)
st.session_state.setdefault("board_y", 9)
st.session_state.setdefault("block_size", 20)  # mm
st.session_state.setdefault("filter_type", "normal")
st.session_state.setdefault("calibration_type", "Regular")
st.session_state.setdefault("session_end_time", 300)

# storage
st.session_state.setdefault("captured_frames", [])
st.session_state.setdefault("captured_names", [])

# permission flag
st.session_state.setdefault("camera_permation", False)

# webrtc ctx store
st.session_state.setdefault("webrtc_ctx", None)


if st.session_state.parameters_page:
    status = st.empty()

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
            "File Name", st.session_state.file_name, max_chars=30, placeholder="file name"
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        st.session_state.board_x = col1.number_input("Board X", 3, 20, int(st.session_state.board_x))
        st.session_state.board_y = col2.number_input("Board Y", 4, 20, int(st.session_state.board_y))
        st.session_state.block_size = col3.number_input("Block size (mm)", 5, 100, int(st.session_state.block_size), step=1)

        col1, col2 = st.columns([1, 1])
        st.session_state.filter_type = col1.selectbox(
            "Select Filter", ["normal", "black&white"],
            index=0 if st.session_state.filter_type == "normal" else 1
        )
        st.session_state.calibration_type = col2.selectbox(
            "Calibration Type", ["Regular", "Medium", "High"],
            index=["Regular", "Medium", "High"].index(st.session_state.calibration_type)
        )

        st.session_state.start_capturing = st.form_submit_button("Start Calibration")

    if st.session_state.start_capturing:
        st.session_state.parameters_page = False
        st.session_state.live_capturing_page = True
        st.session_state.camera_calibration_page = False
        st.session_state.download_page = False
        st.session_state.page_transition = True
        st.rerun()

elif st.session_state.live_capturing_page:
    status = st.empty()

    st.markdown(
        """
        <h3 style='text-align: center; font-weight: 800; margin-bottom: 20px;'>
            Live Camera Capture
        </h3>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,2,1])
    # back button
    st.session_state.back = col1.button("⬅ Back to Parameters")

    if st.session_state.back:
        st.session_state.parameters_page = True
        st.session_state.live_capturing_page = False
        st.session_state.page_transition = False
        st.session_state.camera_calibration_page = False
        st.session_state.download_page = False
        st.session_state.page_transition = True
        st.session_state.start_capturing = False
        st.rerun()

    webrtc_ctx = webrtc_streamer(
        key="video-custom",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,
        video_receiver_size=30,
        sendback_video=False,
    )
    image_place = col2.empty()

    if webrtc_ctx.video_receiver is None:
        status.warning("⚠️ Please allow camera permission from browser popup.")
        st.stop()

    st.session_state.camera_permation = True

    start_time = time.time()
    while True:
        if time.time() - start_time >= st.session_state.session_end_time:
            st.warning("⏳ Stopped because TIME OUT reached.")
            break

        # receiver lost
        if webrtc_ctx.video_receiver is None:
            st.warning("⚠️ Camera disconnected / permission removed.")
            break

        try:
            video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
        except queue.Empty:
            continue
        except Exception as e:
            status.error(f"Error: {e}")
            break

        img_rgb = video_frame.to_ndarray(format="rgb24")
        image_place.image(img_rgb, channels="RGB", use_container_width=True)

        time.sleep(0.08)

# other pages
elif st.session_state.camera_calibration_page:
    st.info("Calibration page will be here")

elif st.session_state.download_page:
    st.info("Download page will be here")

else:
    st.error("this page is empty")
