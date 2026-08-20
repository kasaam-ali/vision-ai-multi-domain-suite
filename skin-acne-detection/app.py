import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Skin Acne Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Skin Acne Detection System")
st.write("Upload a face or skin image to detect acne using the trained YOLO model.")

# Load trained YOLO model with caching
@st.cache_resource
def load_acne_model():
    # Adjust path according to your local folder structure
    return YOLO("models/best.pt")

try:
    model = load_acne_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar options
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.15, step=0.05)

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    if st.button("Detect Acne"):
        with st.spinner("Analyzing image..."):
            # Model prediction
            results = model.predict(source=image, conf=confidence, imgsz=640)
            
            # Plot results (OpenCV BGR array)
            annotated_bgr = results[0].plot()
            
            # BGR to RGB conversion for Streamlit/PIL color fix
            annotated_rgb = annotated_bgr[:, :, ::-1]
            result_img = Image.fromarray(annotated_rgb)
            
            with col2:
                st.subheader("Detection Result")
                st.image(result_img, use_container_width=True)
                
            # Count detected instances
            num_detections = len(results[0].boxes)
            st.success(f"Detection complete! Found {num_detections} acne instance(s).")