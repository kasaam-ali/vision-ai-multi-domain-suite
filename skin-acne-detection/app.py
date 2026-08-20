import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="Skin Acne Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Skin Acne Detection System")
st.markdown("Upload a skin/face image to detect and localize acne spots using YOLOv11.")

# Sidebar Settings
st.sidebar.header("⚙️ Model Settings")
confidence_val = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=1.0,
    value=0.15,
    step=0.05,
    help="Kam confidence faint acne spots ko detect karne ke liye use karein."
)

api_url = st.sidebar.text_input("FastAPI Endpoint", "https://acne-detection-zyrg.onrender.com/predict")

# Main Interface
uploaded_file = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        input_image = Image.open(uploaded_file)
        st.image(input_image, use_container_width=True)
    
    if st.button("🔍 Detect Acne", type="primary", use_container_width=True):
        with st.spinner("Analyzing skin image via FastAPI..."):
            try:
                # Reset file pointer and prepare request
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"confidence": confidence_val}
                
                response = requests.post(api_url, files=files, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    with col2:
                        st.subheader("Detection Result")
                        # Decode base64 image
                        img_bytes = base64.b64decode(data["image_base64"])
                        result_image = Image.open(io.BytesIO(img_bytes))
                        st.image(result_image, use_container_width=True)
                    
                    # Detection Metrics
                    st.success(f"Detections Completed: Found **{data['total_detections']}** acne spot(s).")
                    
                    
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("FastAPI server connect nahi ho saka. Pehle backend start karein: `uvicorn main:app --reload`")
            except Exception as e:
                st.error(f"Error: {str(e)}")
