import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(
    page_title="X-Ray Weapon Detection",
    page_icon="🔫",
    layout="wide"
)

st.title("🔫 X-Ray Weapon Detection System")
st.markdown("Upload an X-Ray baggage image to detect potential weapon threats via FastAPI Microservice.")

# Sidebar Settings
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("FastAPI Endpoint", value="https://vision-ai-multi-domain-suite.onrender.com/predict")

uploaded_file = st.file_uploader("Choose an X-Ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Display Original Image
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    if st.button("Detect Weapons", type="primary"):
        with st.spinner("Analyzing image through backend API..."):
            try:
                # Prepare image payload for FastAPI
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_bytes = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_bytes, "image/png")}
                
                # Call FastAPI endpoint
                response = requests.post(api_url, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get("predictions", [])
                    total_detections = data.get("total_detections", 0)
                    
                    # Draw Bounding Boxes on Image
                    draw_image = image.copy()
                    draw = ImageDraw.Draw(draw_image)
                    
                    for pred in predictions:
                        bbox = pred["bbox"]  # [xmin, ymin, xmax, ymax]
                        label = f"{pred['class_name']} ({pred['confidence']*100:.1f}%)"
                        
                        # Bounding Box Draw
                        draw.rectangle(bbox, outline="red", width=3)
                        # Text Label Draw
                        draw.text((bbox[0], max(0, bbox[1] - 15)), label, fill="red")
                    
                    with col2:
                        st.subheader("Detection Result")
                        st.image(draw_image, use_container_width=True)
                        
                        if total_detections > 0:
                            st.error(f"⚠️ Warning: {total_detections} threat(s) detected!")
                            st.json(predictions)
                        else:
                            st.success("✅ No weapons detected in this scan.")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                st.error(f"Could not connect to FastAPI server: {str(e)}")
