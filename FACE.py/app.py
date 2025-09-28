# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import tempfile
from PIL import Image
import mediapipe as mp
import os

# ---- app layout ----
st.set_page_config(page_title="Face Detection App", layout="wide")
st.title("Face Detection — Realtime, Video & Photo")
st.markdown(
    """
    Simple face detection app using **MediaPipe** + **OpenCV** and served with **Streamlit**.
    - Realtime (webcam)
    - Video file processing
    - Photo/image detection
    """
)

# sidebar
st.sidebar.title("Options")
detection_conf = st.sidebar.slider("Detection confidence", 0.1, 0.99, 0.5, 0.05)
draw_keypoints = st.sidebar.checkbox("Draw keypoints (face landmarks)", value=True)
mode = st.sidebar.radio("Mode", ["Realtime (Webcam)", "Video (Upload)", "Photo (Upload)"])

# prepare mediapipe
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Utility: draw detections on frame (OpenCV image BGR)
def draw_faces_on_frame(frame_bgr, detections, draw_kp=True):
    h, w, _ = frame_bgr.shape
    for det in detections:
        # det.location_data is a NormalizedBoundingBox for face_detection
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = x1 + bw
        y2 = y1 + bh
        # Clamp
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        # rectangle
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # score
        if det.score:
            score = int(det.score[0] * 100)
            cv2.putText(frame_bgr, f"{score}%", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # keypoints (if available)
        if draw_kp and det.location_data.relative_keypoints:
            for kp in det.location_data.relative_keypoints:
                cx = int(kp.x * w)
                cy = int(kp.y * h)
                cv2.circle(frame_bgr, (cx, cy), 3, (0, 200, 255), -1)

    return frame_bgr

# ---------- Realtime (Webcam) Processor ----------
class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=detection_conf)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # convert to RGB for mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(img_rgb)
        if results.detections:
            img = draw_faces_on_frame(img, results.detections, draw_kp=draw_keypoints)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- Main UI ----------

if mode == "Realtime (Webcam)":
    st.header("Realtime Webcam Face Detection")
    st.markdown(
        "Allow camera permission in your browser. Uses `streamlit-webrtc` to access webcam and show detections live."
    )
    # WebRTC config - public STUN server (works in most setups)
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key="face-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
        # optional: display raw html controls or not
    )

elif mode == "Video (Upload)":
    st.header("Process a Video File (detect faces & export)")
    st.markdown("Upload an MP4/AVI file. The app will process and produce an annotated video you can download.")
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_in.write(uploaded_file.read())
        tfile_in.flush()
        cap = cv2.VideoCapture(tfile_in.name)
        if not cap.isOpened():
            st.error("Cannot open uploaded video.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(tfile_out.name, fourcc, fps, (width, height))

            progress = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            processed = 0

            face_detector = mp_face.FaceDetection(min_detection_confidence=detection_conf)

            stframe = st.empty()
            with st.spinner("Processing video — this may take a while..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # process
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = face_detector.process(rgb)
                    if res.detections:
                        frame = draw_faces_on_frame(frame, res.detections, draw_kp=draw_keypoints)
                    out.write(frame)
                    processed += 1
                    if total_frames:
                        progress.progress(min(processed / total_frames, 1.0))
                    # show last frame preview occasionally
                    if processed % max(1, int(fps // 2)) == 0:
                        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            cap.release()
            out.release()
            st.success("Processing finished.")
            st.video(tfile_out.name)
            with open(tfile_out.name, "rb") as f:
                st.download_button("Download processed video", data=f, file_name="processed_faces.mp4", mime="video/mp4")

            # cleanup temp files on disk (optional)
            try:
                os.unlink(tfile_in.name)
            except Exception:
                pass

elif mode == "Photo (Upload)":
    st.header("Photo Face Detection")
    st.markdown("Upload an image (jpg, png). The app will detect faces and display the annotated image.")
    uploaded_img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        # Load image safely with PIL
        image = Image.open(uploaded_img).convert("RGB")

        # Convert to NumPy (RGB) then to BGR for OpenCV
        frame = np.array(image).astype("uint8")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run detection
        face_detector = mp_face.FaceDetection(min_detection_confidence=detection_conf)
        results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.detections:
            frame = draw_faces_on_frame(frame, results.detections, draw_kp=draw_keypoints)

        # Convert back to RGB for display in Streamlit
        display_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show annotated image
        st.image(display_img, channels="RGB", use_column_width=True)

        # Allow download of annotated image
        _, im_buf_arr = cv2.imencode(".png", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        st.download_button(
            "Download annotated image",
            data=im_buf_arr.tobytes(),
            file_name="annotated.png",
            mime="image/png"
        )
