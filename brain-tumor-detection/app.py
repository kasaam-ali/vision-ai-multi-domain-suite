import streamlit as st
import io
import requests
from PIL import Image,ImageDraw,ImageFont

st.set_page_config(page_title="Brain Tumor Detection",page_icon='🧠',layout="wide")
st.title("🧠 Brain Tumor Detection System")
st.markdown('Upload a brain MRI image to detect potential tumor')
st.sidebar.header("configuration")
api_url=st.sidebar.text_input('Fastapi Endpoint',value="http://127.0.0.1:8000/api/v1/detect")
confidence_threshold=st.sidebar.slider("Confidence Threshold",min_value=0.05,max_value=1.0,value=0.15,step=0.05)
uploaded_file=st.file_uploader("choose a brain MRI image",type=['jpg','png','jpeg'])
if uploaded_file is not None:
    col1,col2=st.columns(2)
    image=Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("Original Image")
        st.image(image,use_container_width=True)
        if st.button("Detect Tumor"):
            with st.spinner("Analyzing image through backend API..."):
                 img_byte_arr = io.BytesIO()
                 image.save(img_byte_arr, format='PNG')
                 img_bytes=img_byte_arr.getvalue()
                 files={'file':(uploaded_file.name,img_bytes,'image/png')}
                 response=requests.post(api_url,files=files)
                 if response.status_code==200:
                     data=response.json()
                     predictions=data.get('predictions',[])
                     total_detections=data.get('total_detections',0)
                     draw_image=image.copy()
                     draw = ImageDraw.Draw(draw_image)
                     for pred in predictions:
                         bbox=pred['bbox']
                         label=f'{pred['class_name']}: {pred['confidence']:.2f}'
                         draw.rectangle(bbox,outline='red',width=3)
                         draw.text((bbox[0],max(0,bbox[1]-15)),label,fill='red')
                     
                     with col2:
                         st.subheader('Detection Result')
                         st.image(draw_image,use_container_width=True)
                         st.write(f'Total Detections: {total_detections}')
