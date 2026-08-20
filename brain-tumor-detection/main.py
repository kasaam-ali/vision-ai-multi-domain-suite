from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import numpy as np 
import cv2
from ultralytics import YOLO
from PIL import Image 
import io 
import os

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "models/best.pt" if os.path.exists("models/best.pt") else "best.pt"
    print(f"Loading Brain Tumor Detection Model from {model_path}...")
    try:
        models["brain_tumor"] = YOLO(model_path)
        print("Brain Tumor Detection Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
    yield
    models.clear()
    print("Model resources cleared.")

app = FastAPI(
    title="Brain Tumor Detection API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/heath")
def health_check():
    return {"status": "online", "message": "API is Working"}

@app.post("/api/v1/detect")
async def predict_tumor(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    
    if "brain_tumor" not in models:
        raise HTTPException(status_code=500, detail="Brain tumor model is not loaded.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        model = models["brain_tumor"]
        results = await run_in_threadpool(model.predict, image_np, conf=0.25, imgsz=640)
        
        predictions = []
        for box in results[0].boxes:
            predictions.append({
                'bbox': box.xyxy[0].tolist(),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': models['brain_tumor'].names[int(box.cls[0])]
            })
        
        return {
            "status": "success",
            "filename": file.filename,
            "total_detections": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")