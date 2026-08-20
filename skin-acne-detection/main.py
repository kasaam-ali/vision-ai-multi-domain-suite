from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI(
    title="Skin Acne Detection API",
    description="Production-ready FastAPI backend for YOLOv11 Acne Detection",
    version="1.0"
)

# Model load on startup
MODEL_PATH = "models/best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "online", "model": "YOLOv11 Acne Detector"}

@app.post("/predict")
async def predict_acne(
    file: UploadFile = File(...),
    confidence: float = Query(0.15, ge=0.01, le=1.0, description="Confidence threshold")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Inference
        results = model.predict(source=image, conf=confidence, imgsz=640)
        result = results[0]
        
        # Extract structured prediction data
        detections = [
            {
                "bbox": [round(coord, 2) for coord in box.xyxy[0].tolist()],
                "confidence": round(float(box.conf[0]), 4),
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])]
            }
            for box in result.boxes
        ]
        
        # Generate plotted image in RGB
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)
        
        # Convert annotated image to Base64
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "success": True,
            "total_detections": len(detections),
            "detections": detections,
            "image_base64": encoded_image
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))