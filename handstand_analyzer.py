import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Handstand Analyzer", page_icon="🤸", layout="wide")

st.title("🤸 Handstand Analyzer")
st.write("Upload a handstand image to analyze your form")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    detection_confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)
    tracking_confidence = st.slider("Tracking Confidence", 0.0, 1.0, 0.5, 0.05)

uploaded_file = st.file_uploader("Upload handstand image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file)
    
    # Process image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=detection_confidence,
    ) as pose:
        results = pose.process(image_rgb)
        
        with col2:
            st.subheader("Pose Detection")
            if results.pose_landmarks:
                # Draw landmarks
                annotated_image = image_rgb.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                st.image(annotated_image)
                st.success("✅ Pose detected successfully!")
                
            else:
                st.error("❌ No pose detected. Try a clearer image.")
else:
    st.info("👆 Upload an image to get started")