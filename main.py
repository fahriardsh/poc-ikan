from fastapi import FastAPI, Response, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import base64
import time
from ultralytics import YOLO
import numpy as np
from typing import Optional, List
import io
import torch

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load YOLO model
try:
    model = YOLO("my_model/ikan-78/my_model.pt")
    # Check if CUDA is available and model is using GPU
    if torch.cuda.is_available():
        device = "GPU"
        model.to('cuda')
    else:
        device = "CPU"
    print(f"Model loaded successfully on {device}")
except FileNotFoundError:
    print("Model file not found. Using a placeholder model.")
    model = None
    device = "Unknown"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect_fish(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    
    # Run detection
    print(f"Running detection on {device}")
    results = model(img)
    annotated_frame = results[0].plot()
    
    # Process results
    detections = []
    detected_class = None
    max_conf = 0
    
    if len(results[0].boxes) > 0:
        for r in results[0].boxes:
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            label = model.names[cls]
            
            # Add to detections list
            detections.append({
                "class": label,
                "confidence": conf
            })
            
            # Track the highest confidence detection
            if conf > max_conf:
                max_conf = conf
                detected_class = label
    
    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare response
    if detected_class and max_conf > 0.6:  # Confidence threshold
        result = {
            "status": "success",
            "result": f"{detected_class} ({max_conf:.1%})",
            "image": jpg_as_text,
            "confidence": max_conf,
            "detections": detections,
            "device": device
        }
    else:
        # Convert original image to base64 (no detection)
        _, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        result = {
            "status": "success",
            "result": "No fish detected",
            "image": jpg_as_text,
            "confidence": 0,
            "detections": [],
            "device": device
        }
    
    return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
