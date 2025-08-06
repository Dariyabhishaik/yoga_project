# streamlit_app.py
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf

st.set_page_config(page_title="Yoga Pose Corrector", layout="centered")
st.title("🧘‍♀️ AI Yoga Pose Detector and Corrector")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load TensorFlow model and label map
model = tf.keras.models.load_model("pose_model.h5")
label_map = {0: "Tree Pose", 1: "Warrior Pose"}

# Helper: Get 33 landmarks (x, y)
def extract_landmark_array(landmarks):
    keypoints = []
    for lm in landmarks:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    return np.array(keypoints).reshape(1, -1)

# TensorFlow classification
def classify_pose_with_model(landmarks):
    input_data = extract_landmark_array(landmarks)
    pred = model.predict(input_data)[0]
    label_idx = np.argmax(pred)
    confidence = pred[label_idx]
    return label_map[label_idx], confidence > 0.8

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
            pose_name, is_correct = classify_pose_with_model(results.pose_landmarks.landmark)

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
                pose_name, is_correct = classify_pose_with_model(results.pose_landmarks.landmark)
                label = f"{pose_name} - {'Correct' if is_correct else 'Incorrect'}"
                cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if is_correct else (0, 0, 255), 2, cv2.LINE_AA)
            return image

    st.warning("Make sure your webcam is enabled and accessible.")
    webrtc_streamer(key="yoga", video_transformer_factory=VideoTransformer)

