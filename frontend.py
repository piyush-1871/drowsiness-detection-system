import cv2
import numpy as np
import dlib
from imutils import face_utils
import streamlit as st
import time

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Reusable function to compute distance
def compute(ptA, ptB):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)

# Reusable function to determine blink status
def blinked(a, b, c, d, e, f):
    """Calculate the eye blink ratio and return the blink status."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Eye Open
    elif ratio > 0.21 and ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Sleeping


# Title
st.title("Drowsiness Detection System!")

# Menu options
choice = st.sidebar.selectbox("My Menu", ("Home", "Web Cam"))

if choice == "Home":
    st.header("Welcome")
    st.write("This app detects the drowsiness of a person in the video. Built with OpenCV and DLib.")

elif choice == "Web Cam":
    # Create placeholder for webcam feed (dynamic)
    frame_placeholder = st.empty()

    # Define the width of the frame dynamically
    frame_width = 640  # Define the frame width (you can set your own width here)
    frame_height = 480  # Set a fixed height for the frame

    # Create columns for the Start and Stop buttons
    col1, col2 = st.columns([frame_width / 2, frame_width / 2])

    with col1:
        start_button = st.button('Start')
    with col2:
        stop_button = st.button('Stop')

    if start_button:
        # Start button pressed, begin webcam and detection
        cap = cv2.VideoCapture(0)

        # Set the desired frame width and height for the webcam feed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        # Status marking for the current state
        sleep = 0
        drowsy = 0
        active = 0
        status = ""
        color = (0, 0, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Detected face in faces array
            for face in faces:
                x1 = face.left()
                x2 = face.right()
                y1 = face.top()
                y2 = face.bottom()

                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # The numbers are actually the landmarks which will show the eye
                left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

                # Now judge what to do for the eye blinks
                if left_blink == 0 and right_blink == 0:
                    sleep += 1
                    drowsy = 0
                    active = 0
                    if sleep > 6:
                        status = "SLEEPING!!"
                        color = (255, 0, 0)  # Red
                elif left_blink == 1 or right_blink == 1:
                    sleep = 0
                    active = 0
                    drowsy += 1
                    if drowsy > 6:
                        status = "DROWSY!!"
                        color = (0, 0, 255)  # Blue
                else:
                    drowsy = 0
                    sleep = 0
                    active += 1
                    if active > 6:
                        status = "ACTIVE :)"
                        color = (0, 255, 0)  # Green

                # Draw the rectangle around the face with the dynamic color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Display status on the frame
            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Convert frame to RGB before displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Show the main frame with the rectangle and status
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Pause to control the frame rate and prevent excessive loading
            time.sleep(0.1)  # Delay for ~10 FPS

            if stop_button:
                # Stop button pressed, release webcam and break the loop
                cap.release()
                cv2.destroyAllWindows()
                break
    else:
        st.write("Press 'Start' to begin the drowsiness detection.")
