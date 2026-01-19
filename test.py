# col1, col2 = st.columns([1,1])
# Placeholder for live feed
# frame_placeholder11 = col1.empty()
# frame_placeholder12 = col2.empty()
# frame_placeholder21 = col1.empty()
# frame_placeholder22 = col2.empty()


# h, w = frame.shape[:2]      # h=480, w=640

# half_w = w // 2             # 320
# half_h = h // 2             # 240

# frame1 = frame[0:half_h, 0:half_w]
# frame2 = frame[0:half_h, half_w:w]           # Top-Right
# frame3 = frame[half_h:h, 0:half_w]           # Bottom-Left
# frame4 = frame[half_h:h, half_w:w]  

# frame_placeholder11.image(frame1, channels="RGB")
# frame_placeholder12.image(frame2, channels="RGB")
# frame_placeholder21.image(frame3, channels="RGB")
# frame_placeholder22.image(frame4, channels="RGB")



        # if counter%100==0:
        #     first_corner[0]+=x_add
        #     second_corner[0]+=x_add
        #     if second_corner[0] >= frame_size[0]:
        #         first_corner[1]+=y_add
        #         second_corner[1]+=y_add
        #         first_corner[0] = 0
        #         second_corner[0] = 64
        #     if second_corner[1] >= frame_size[1]:
        #         first_corner = [0, 0]
        #         second_corner = [64, 48]

import logging
import queue
import time
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Custom Camera UI", layout="wide")
st.title("📷 Camera Feed (Custom UI)")

# ---------------------------
# Session state
# ---------------------------
st.session_state.setdefault("run_camera", False)

# ---------------------------
# Custom buttons
# ---------------------------
col1, col2 = st.columns([1, 1])
start_btn = col1.button("▶ Start Camera", use_container_width=True)
stop_btn = col2.button("⛔ Stop Camera", use_container_width=True)

if start_btn:
    st.session_state.run_camera = True
if stop_btn:
    st.session_state.run_camera = False

# ---------------------------
# ✅ WebRTC controlled by custom state
# ---------------------------
webrtc_ctx = webrtc_streamer(
    key="video-custom",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"video": True, "audio": False},
    desired_playing_state=st.session_state.run_camera,
    video_receiver_size=30,   # ✅ FIX queue overflow
    sendback_video=False,
)

status = st.empty()
image_place = st.empty()

# ---------------------------
# Display loop
# ---------------------------
if not st.session_state.run_camera:
    status.info("Click ▶ Start Camera to begin.")
else:
    if webrtc_ctx.video_receiver is None:
        status.warning("⚠️ Please allow camera permission first.")
    else:
        status.success("✅ Camera running")

        # ✅ SAFE LOOP: only run 2 seconds then rerun
        t0 = time.time()

        while time.time() - t0 < 2:
            if not st.session_state.run_camera:
                break

            if webrtc_ctx.video_receiver is None:
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

            # ✅ limit FPS (very important)
            time.sleep(0.08)   # ~12 FPS

        # ✅ refresh streamlit script
        st.rerun()
