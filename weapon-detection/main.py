from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

# Dictionary to hold model instance in memory
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load YOLOv11 model into memory once when application starts
    models["weapon_model"] = YOLO("models/best.pt")
    print("YOLOv11 Weapon Detection Model Loaded Successfully!")
    yield
    # Shutdown: Clean up resources on app exit
    models.clear()
    print("Model resources cleared.")

app = FastAPI(
    title="X-Ray Weapon Detection API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    return {"status": "online", "message": "Weapon Detection Service is running"}

@app.post("/predict")
@app.post("/api/v1/detect")
async def detect_weapons(file: UploadFile = File(...)):
    # Validate uploaded file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # Read image bytes asynchronously
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Offload CPU/GPU heavy inference to threadpool to avoid blocking event loop
        model = models["weapon_model"]
        results = await run_in_threadpool(model.predict, image_np, conf=0.25,imgsz=640)
        
        predictions = []
        for result in results:
            for box in result.boxes:
                # Extract coordinates and cast to pure Python float (Option B: JSON Payload)
                coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                predictions.append({
                    "bbox": [round(c, 2) for c in coords],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name
                })
                
        return {
            "filename": file.filename,
            "total_detections": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")