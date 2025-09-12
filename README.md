# poc-ikan

## Fish Detection with Webcam using FastAPI

This project demonstrates fish detection using a client-side webcam, YOLO model, and FastAPI backend.

### Features

- Client-side webcam access (browser-based)
- Capture images and send to backend for fish detection
- Real-time fish detection using YOLO model
- Web interface with start/capture/reset controls

### Installation

1. Install dependencies using UV:
   ```bash
   uv install
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

### Usage

1. Run the FastAPI application:
   ```bash
   uv run main.py
   ```

   Or:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to `http://localhost:3000`

3. Click "Start Webcam" to access your camera

4. Click "Capture & Detect" to capture an image and send it to the backend for fish detection

5. View the detection results with annotated image

### API Endpoints

- `GET /` - Serve the main web interface
- `POST /detect` - Receive image and return detection results

### Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- OpenCV - Computer vision library
- Ultralytics - YOLO implementation
- *.pt - for the pretrained model