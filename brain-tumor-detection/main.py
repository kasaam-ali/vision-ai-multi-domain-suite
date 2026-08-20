from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import io
import base64

app = FastAPI(
    title="Brain Tumor Detection API",
    description="FastAPI Backend for YOLOv11 Brain Tumor MRI Detection",
    version="1.0"
)

# Robust Path Resolution: checks for best.pt or best (4).pt automatically
CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "models"

model_file = None
for candidate in ["best.pt", "best (4).pt"]:
    p = MODELS_DIR / candidate
    if p.exists():
        model_file = p
        break

if not model_file:
    # Check root models directory as fallback
    fallback = Path("models/best.pt")
    if fallback.exists():
        model_file = fallback
    else:
        raise FileNotFoundError(f"Model file not found in {MODELS_DIR}. Please place 'best.pt' inside models folder.")

# Load model
model = YOLO(str(model_file))
print(f"Loaded YOLO Model from: {model_file}")
print(f"Classes: {model.names}")


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "YOLOv11 Brain Tumor Detector",
        "classes": model.names
    }


@app.post("/predict")
async def predict_tumor(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sirf image files upload karein.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # YOLOv11 Inference
        results = model.predict(source=image, conf=confidence, imgsz=640)
        result = results[0]

        # Extract Detections
        detections = [
            {
                "bbox": [round(coord, 2) for coord in box.xyxy[0].tolist()],
                "confidence": round(float(box.conf[0]), 4),
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])]
            }
            for box in result.boxes
        ]

        # Convert BGR Plot to RGB
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)

        # Convert to Base64
        buffer = io.BytesIO()
        annotated_pil.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "tumor_detected": bool(len(detections) > 0),
            "total_detections": len(detections),
            "detections": detections,
            "image_base64": encoded_image
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")