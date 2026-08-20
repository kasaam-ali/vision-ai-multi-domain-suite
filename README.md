# Vision AI Multi-Domain Suite

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv11-Object%20Detection-111827?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv11">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-3%20Domains-7C3AED?style=flat-square" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Inference-Local%20%26%20Private-16A34A?style=flat-square" alt="Local inference">
  <img src="https://img.shields.io/badge/License-Educational%20Use-F59E0B?style=flat-square" alt="Educational use">
</p>

<p align="center">
  <strong>One computer vision workspace. Three focused detection applications.</strong>
</p>

<p align="center">
  🧠 Brain Tumor MRI &nbsp;•&nbsp; 🩺 Skin Acne &nbsp;•&nbsp; 🛡️ X-Ray Weapon Detection
</p>

---

## Overview

**Vision AI Multi-Domain Suite** is a collection of independent computer vision applications powered by custom YOLOv11 models. Each domain includes:

- A **FastAPI inference backend** for model loading and prediction
- A **Streamlit interface** for image upload and visual results
- Bounding-box localization with confidence scores
- A lightweight JSON API for integration with other clients
- Local model inference, keeping uploaded images inside the local application environment

The applications are intentionally separated so each model can be developed, tested, and deployed independently.

> ⚠️ **Important:** This project is for educational, research, and prototyping purposes. Brain tumor and acne results are not medical diagnoses, and weapon results are not a replacement for trained security personnel or certified screening systems.

## Applications

| Application | Input | Detection | Backend endpoint | UI |
|---|---|---|---|---|
| 🧠 Brain Tumor Detection | Brain MRI: JPG, JPEG, PNG | Tumor regions, class, confidence, annotated image | `POST /predict` | Streamlit |
| 🩺 Skin Acne Detection | Skin or face image: JPG, JPEG, PNG | Acne spots, class, confidence, annotated image | `POST /predict` | Streamlit |
| 🛡️ Weapon Detection | X-ray baggage image: JPG, JPEG, PNG | Potential weapon regions and confidence | `POST /api/v1/detect` | Streamlit |

## Architecture

```text
Image upload
     │
     ▼
Streamlit UI  ───── multipart/form-data ─────►  FastAPI API
                                                     │
                                                     ▼
                                              YOLOv11 model
                                                     │
                                                     ▼
                         JSON detections + annotated image / boxes
```

Each folder is a standalone service:

```text
vision-ai-multi-domain-suite/
├── brain-tumor-detection/
│   ├── app.py                 # Streamlit UI
│   ├── main.py                # FastAPI API
│   ├── models/best (4).pt     # YOLOv11 weights
│   └── requirements.txt
├── skin-acne-detection/
│   ├── app.py                 # Streamlit UI
│   ├── main.py                # FastAPI API
│   ├── models/best.pt         # YOLOv11 weights
│   └── requirements.txt
├── weapon-detection/
│   ├── app.py                 # Streamlit UI
│   ├── main.py                # FastAPI API
│   ├── models/best.pt         # YOLOv11 weights
│   └── requirements.txt
├── requirements.txt           # Shared environment dependencies
└── .gitignore
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Git
- A CPU or CUDA-compatible GPU
- A modern web browser
- The matching `best.pt` file inside each application's `models/` directory

> 💡 CPU inference works, but a compatible NVIDIA GPU can significantly improve prediction speed.

## Installation

From the repository root, create and activate the existing virtual environment or create a new one:

### Windows PowerShell

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Command Prompt

```bat
myenv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The shared `requirements.txt` contains the core FastAPI, Streamlit, Ultralytics, image-processing, and numerical dependencies used by the suite.

## Running an Application

Each application needs **two terminals**: one for FastAPI and one for Streamlit. Run the commands from the selected application directory.

### 🧠 Brain Tumor Detection

**Terminal 1: API**

```powershell
cd brain-tumor-detection
uvicorn main:app --reload --port 8000
```

**Terminal 2: UI**

```powershell
cd brain-tumor-detection
streamlit run app.py
```

Open the Streamlit URL shown in the terminal, usually `http://localhost:8501`.

### 🩺 Skin Acne Detection

**Terminal 1: API**

```powershell
cd skin-acne-detection
uvicorn main:app --reload --port 8000
```

**Terminal 2: UI**

```powershell
cd skin-acne-detection
streamlit run app.py
```

### 🛡️ Weapon Detection

**Terminal 1: API**

```powershell
cd weapon-detection
uvicorn main:app --reload --port 8000
```

**Terminal 2: UI**

```powershell
cd weapon-detection
streamlit run app.py
```

> ⚠️ Run one backend on port `8000` at a time with the default UI settings. To run multiple domains simultaneously, start them on separate ports and update the **FastAPI Endpoint** field in the relevant Streamlit sidebar.

Example:

```powershell
uvicorn main:app --reload --port 8002
```

Then set the UI endpoint to `http://127.0.0.1:8002/predict` for the brain tumor or acne application, or `http://127.0.0.1:8002/api/v1/detect` for weapon detection.

## API Reference

### Health checks

```http
GET /
```

Example response:

```json
{
  "status": "online"
}
```

### Brain tumor and acne prediction

```http
POST /predict
Content-Type: multipart/form-data
```

Parameters:

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `file` | image file | required | JPG, JPEG, or PNG image |
| `confidence` | float | `0.25` brain / `0.15` acne | Value between `0.01` and `1.0` |

Example with `curl`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?confidence=0.25" `
  -F "file=@sample-mri.jpg"
```

Typical response fields:

```json
{
  "success": true,
  "tumor_detected": true,
  "total_detections": 1,
  "detections": [
    {
      "bbox": [120.5, 80.2, 310.7, 260.9],
      "confidence": 0.9342,
      "class_id": 0,
      "class_name": "tumor"
    }
  ],
  "image_base64": "..."
}
```

The acne endpoint returns the same core detection structure with `total_detections`, `detections`, and `image_base64`.

### Weapon prediction

```http
POST /api/v1/detect
Content-Type: multipart/form-data
```

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/detect" `
  -F "file=@sample-xray.png"
```

Typical response:

```json
{
  "filename": "sample-xray.png",
  "total_detections": 1,
  "predictions": [
    {
      "bbox": [90.0, 120.0, 280.0, 360.0],
      "confidence": 0.8765,
      "class_id": 0,
      "class_name": "weapon"
    }
  ]
}
```

Interactive API documentation is available while a FastAPI service is running:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Confidence Thresholds

The confidence threshold controls how selective the detector is:

- **Lower threshold:** may detect faint objects, but can produce more false positives
- **Higher threshold:** produces fewer, more confident detections, but can miss subtle objects

Use the Streamlit sidebar to adjust this value for the image and model being tested.

## Troubleshooting

### `FastAPI server offline` or connection refused

Start the backend first and confirm that the UI endpoint matches the backend port and path.

```powershell
uvicorn main:app --reload --port 8000
```

### Model file not found

Confirm that the required weights are present:

```text
brain-tumor-detection/models/best (4).pt
skin-acne-detection/models/best.pt
weapon-detection/models/best.pt
```

The brain tumor backend also supports the alternate filename `best (4).pt`.

### Port already in use

Start the backend on another port and update the endpoint in the Streamlit sidebar:

```powershell
uvicorn main:app --reload --port 8002
```

### Dependency or import errors

Activate the correct virtual environment, then reinstall the shared dependencies:

```powershell
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Slow predictions

The first prediction can be slower because the model and runtime initialize. For repeated inference, use a GPU-enabled PyTorch installation that matches your CUDA version.

## Development Notes

- Models are loaded once by the FastAPI service and reused for predictions.
- Uploaded images are processed in memory and are not intentionally persisted by the application.
- Brain tumor and acne APIs return an annotated image as a Base64-encoded JPEG.
- Weapon detection returns structured bounding boxes, while the Streamlit client renders those boxes locally.
- Do not commit virtual environments, Python caches, or compiled files; these are covered by `.gitignore`.

## Responsible Use

This repository demonstrates applied computer vision and API integration. It must not be used as the sole basis for:

- Medical diagnosis or treatment decisions
- Airport, hospital, or public security decisions
- Legal or disciplinary action
- Automated decisions affecting a person's rights or safety

Validate model performance on representative data, monitor false positives and false negatives, protect uploaded images, and keep a qualified human in the decision loop.

## License

No open-source license has been declared yet. Until a license is added, treat the repository as **all rights reserved** and use it only with the author's permission.

---

<p align="center">
  Built with 🐍 Python, ⚡ FastAPI, 🎈 Streamlit, and 🤖 YOLOv11
</p>
