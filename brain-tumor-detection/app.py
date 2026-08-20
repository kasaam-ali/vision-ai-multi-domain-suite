import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="Brain Tumor MRI Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection System")
st.markdown("Upload a Brain MRI scan to localize and classify brain tumors using YOLOv11.")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
confidence_val = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Adjust threshold to detect faint tumor regions."
)

api_url = st.sidebar.text_input("FastAPI Endpoint", "https://brain-tumor-dau6.onrender.com/predict")

# Main UI
uploaded_file = st.file_uploader("Upload Brain MRI Scan (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI Scan")
        input_image = Image.open(uploaded_file)
        st.image(input_image, use_container_width=True)

    if st.button("🔬 Analyze MRI Scan", type="primary", use_container_width=True):
        with st.spinner("Processing MRI scan via FastAPI..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"confidence": confidence_val}

                response = requests.post(api_url, files=files, params=params)

                if response.status_code == 200:
                    data = response.json()

                    total_count = data.get("total_detections", len(data.get("detections", [])))
                    has_tumor = data.get("tumor_detected", total_count > 0)
                    encoded_img = data.get("image_base64", None)
                    detections = data.get("detections", [])

                    with col2:
                        st.subheader("Detection Result")
                        if encoded_img:
                            img_bytes = base64.b64decode(encoded_img)
                            result_image = Image.open(io.BytesIO(img_bytes))
                            st.image(result_image, use_container_width=True)

                    if has_tumor:
                        st.error(f"⚠️ **Tumor Detected!** Total Regions Found: **{total_count}**")
                    else:
                        st.success("✅ **No Tumor Detected** at this confidence threshold.")

                    if total_count > 0:
                        with st.expander("📋 View Bounding Boxes & Class Predictions"):
                            st.json(detections)
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ FastAPI Server offline hai. Terminal mein pehle backend start karein: `uvicorn main:app --reload`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
