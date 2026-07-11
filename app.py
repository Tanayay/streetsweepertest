import base64
import io
import os
from functools import lru_cache

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}


@lru_cache(maxsize=1)
def get_model():
    model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
    return YOLO(model_name)


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/detect")
def detect():
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "Choose an image first."}), 400

    if uploaded.mimetype not in ALLOWED_TYPES:
        return jsonify({"error": "Use a JPG, PNG, or WEBP image."}), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "That file is not a readable image."}), 400

    confidence = request.form.get("confidence", "0.35")
    try:
        confidence_value = min(max(float(confidence), 0.1), 0.9)
    except ValueError:
        confidence_value = 0.35

    model = get_model()
    results = model.predict(
        source=image,
        conf=confidence_value,
        imgsz=960,
        verbose=False,
    )

    result = results[0]
    vehicle_count = 0
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        score = float(box.conf[0].item())

        if class_name not in VEHICLE_NAMES:
            continue

        vehicle_count += 1
        x1, y1, x2, y2 = [round(value, 1) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "type": class_name,
                "confidence": round(score, 3),
                "box": [x1, y1, x2, y2],
            }
        )

    annotated_array = result.plot(
        labels=True,
        conf=True,
        boxes=True,
        pil=False,
    )
    annotated_rgb = annotated_array[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    return jsonify(
        {
            "count": vehicle_count,
            "detections": detections,
            "annotated_image": image_to_data_url(annotated_image),
            "note": "This prototype counts visible vehicles. A curbside parking-zone filter is the next accuracy upgrade.",
        }
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 15 MB."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
