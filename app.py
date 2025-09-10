from flask import Flask, render_template, Response, jsonify
import cv2, time, base64
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("my_model/my_model.pt")

# Kamera laptop
camera = cv2.VideoCapture(0)

# Global variables
result_data = {"status": "scanning", "result": "Waiting...", "image": ""}
confidence_start_time = None
CONF_THRESHOLD = 0.8
HOLD_DURATION = 3  # detik

def generate_frames():
    global result_data, confidence_start_time
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # default
        current_result = "Not found"
        max_conf = 0
        detected_class = None

        for r in results[0].boxes:
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            label = model.names[cls]
            if conf > max_conf:
                max_conf = conf
                detected_class = label

        if detected_class and max_conf > CONF_THRESHOLD:
            if confidence_start_time is None:
                confidence_start_time = time.time()  # mulai hitung
            elapsed = time.time() - confidence_start_time

            current_result = f"{detected_class} ({max_conf:.1%})"
            if elapsed >= HOLD_DURATION:
                # capture image
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                result_data = {
                    "status": "done",
                    "result": f"{detected_class} ({max_conf:.1%})",
                    "image": jpg_as_text
                }
                break
        else:
            # reset jika confidence turun
            confidence_start_time = None
            current_result = "Not found"

        result_data["result"] = current_result

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    return jsonify(result_data)


@app.route('/reset')
def reset():
    global result_data, confidence_start_time
    result_data = {"status": "scanning", "result": "Waiting...", "image": ""}
    confidence_start_time = None
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
