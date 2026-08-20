from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import cv2
import numpy as np
from PIL import Image
import io
import os
from ultralytics import YOLO

# Storage dictionary for global model state
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Acne Detection Model
    model_path = "models/best.pt" if os.path.exists("models/best.pt") else "best.pt"
    print(f"Loading Acne Detection Model from {model_path}...")
    try:
        models["acne_model"] = YOLO(model_path)
        print("Acne Detection Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
    yield
    # Shutdown: Clean up resources
    models.clear()
    print("Model resources cleared.")

app = FastAPI(
    title="Skin Acne Detection API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    return {"status": "online", "message": "Skin Acne Detection Service is running"}


@app.post("/image/detect")
async def detect_acne(file: UploadFile = File(...)):
    # Validate uploaded file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    if "acne_model" not in models:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
    
    try:
        # Read image bytes asynchronously
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Offload heavy compute to threadpool
        model = models["acne_model"]
        results = await run_in_threadpool(model.predict, image_np, conf=0.25, imgsz=640)
        
        predictions = []
        for box in results[0].boxes:
            predictions.append({
                'bbox': box.xyxy[0].tolist(),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': models['acne_model'].names[int(box.cls[0])]
            })
                
        return {
            "status": "success",
            "filename": file.filename,
            "total_detections": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")