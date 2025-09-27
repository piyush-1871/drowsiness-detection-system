import cv2
import numpy as np
import dlib
from imutils import face_utils
import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==========================
# Load face detector & predictor
# ==========================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ==========================
# Utility functions
# ==========================
def compute(ptA, ptB):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    """Calculate eye blink ratio and return status."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Eye Open
    elif 0.21 < ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Sleeping

# ==========================
# WebRTC Video Transformer
# ==========================
class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.sleep = 0
        self.drowsy = 0
        self.active = 0
        self.status = ""
        self.color = (0, 0, 0)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, x2, y1, y2 = face.left(), face.right(), face.top(), face.bottom()
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                                 landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                                  landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 and right_blink == 0:
                self.sleep += 1
                self.drowsy = 0
                self.active = 0
                if self.sleep > 6:
                    self.status = "SLEEPING!!"
                    self.color = (255, 0, 0)
            elif left_blink == 1 or right_blink == 1:
                self.sleep = 0
                self.active = 0
                self.drowsy += 1
                if self.drowsy > 6:
                    self.status = "DROWSY!!"
                    self.color = (0, 0, 255)
            else:
                self.drowsy = 0
                self.sleep = 0
                self.active += 1
                if self.active > 6:
                    self.status = "ACTIVE :)"
                    self.color = (0, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)

        cv2.putText(img, self.status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)
        return img

# ==========================
# Streamlit UI
# ==========================
st.title("😴 Drowsiness Detection System")

choice = st.sidebar.selectbox("Menu", ("Home", "Web Cam"))

if choice == "Home":
    st.header("Welcome")
    st.write("This app detects drowsiness using facial landmarks. "
             "Built with OpenCV, DLib, and Streamlit.")

elif choice == "Web Cam":
    st.write("Turn on your webcam below 👇")
    webrtc_streamer(
        key="drowsiness",
        video_processor_factory=DrowsinessDetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
