# streamlit_app.py
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Yoga Pose Corrector", layout="centered")
st.title("🧘‍♀️ AI Yoga Pose Detector and Corrector")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Helper function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Tree Pose validation
def check_tree_pose(landmarks):
    try:
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        angle = calculate_angle(left_hip, left_knee, left_ankle)
        st.write(f"Tree Pose Leg Angle: {int(angle)}°")
        return 40 < angle < 70
    except:
        return False

# Warrior Pose validation (sample logic)
def check_warrior_pose(landmarks):
    try:
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        st.write(f"Warrior Pose Arm Angle: {int(angle)}°")
        return 150 < angle < 180
    except:
        return False

# Unified pose classifier
def classify_pose(landmarks):
    if check_tree_pose(landmarks):
        return "Tree Pose", True
    elif check_warrior_pose(landmarks):
        return "Warrior Pose", True
    else:
        return "Unknown Pose", False

# Upload or Webcam mode
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose a yoga image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pose_name, is_correct = classify_pose(results.pose_landmarks.landmark)

            st.image(image_rgb, caption="Detected Pose", use_column_width=True)
            st.subheader(f"Pose: {pose_name}")

            if is_correct:
                st.success("✅ Your pose looks correct!")
            else:
                st.error("❌ Your pose seems incorrect.")
                if pose_name == "Tree Pose":
                    st.image("correct_poses/tree_pose.png", caption="Correct Tree Pose")
                elif pose_name == "Warrior Pose":
                    st.image("correct_poses/warrior_pose.png", caption="Correct Warrior Pose")

elif option == "Use Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                pose_name, is_correct = classify_pose(results.pose_landmarks.landmark)
                label = f"{pose_name} - {'Correct' if is_correct else 'Incorrect'}"
                cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_correct else (0, 0, 255), 2, cv2.LINE_AA)
            return image

    st.warning("Make sure your webcam is enabled and accessible.")
    webrtc_streamer(key="yoga", video_transformer_factory=VideoTransformer)
