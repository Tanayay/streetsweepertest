import base64
import io
import os
from functools import lru_cache

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}


@lru_cache(maxsize=1)
def get_model():
    return YOLO(os.getenv("YOLO_MODEL", "yolov8n.pt"))


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0


def nms(detections, threshold=0.45):
    kept = []
    for detection in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if all(iou(detection["box"], existing["box"]) < threshold for existing in kept):
            kept.append(detection)
    return kept


def collect_detections(model, source, confidence, offset_x=0, offset_y=0, imgsz=960):
    detections = []
    result = model.predict(source=source, conf=confidence, imgsz=imgsz, verbose=False)[0]
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if class_name not in VEHICLE_NAMES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "type": class_name,
            "confidence": float(box.conf[0].item()),
            "box": [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
        })
    return detections


def tiled_detect(image, model, confidence):
    width, height = image.size
    detections = collect_detections(model, image, confidence, imgsz=1280)

    tile_size = 640
    overlap = 160
    step = tile_size - overlap
    if width > tile_size or height > tile_size:
        xs = list(range(0, max(width - tile_size, 0) + 1, step))
        ys = list(range(0, max(height - tile_size, 0) + 1, step))
        if not xs or xs[-1] != max(width - tile_size, 0):
            xs.append(max(width - tile_size, 0))
        if not ys or ys[-1] != max(height - tile_size, 0):
            ys.append(max(height - tile_size, 0))

        for y in ys:
            for x in xs:
                crop = image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
                detections.extend(collect_detections(model, crop, confidence, x, y, imgsz=960))

    return nms(detections)


def annotate(image, detections):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 350))
    for detection in detections:
        box = [round(v) for v in detection["box"]]
        label = f'{detection["type"]} {detection["confidence"]:.2f}'
        draw.rectangle(box, outline=(0, 180, 255), width=line_width)
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle((text_box[0] - 3, text_box[1] - 2, text_box[2] + 3, text_box[3] + 2), fill=(0, 45, 85))
        draw.text((box[0], box[1]), label, fill="white", font=font)
    return output


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

    try:
        confidence = min(max(float(request.form.get("confidence", "0.25")), 0.10), 0.90)
        model = get_model()
        detections = tiled_detect(image, model, confidence)
        annotated = annotate(image, detections)
    except Exception as exc:
        app.logger.exception("Detection failed")
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    clean = [
        {"type": d["type"], "confidence": round(d["confidence"], 3), "box": [round(v, 1) for v in d["box"]]}
        for d in detections
    ]
    return jsonify({
        "count": len(clean),
        "detections": clean,
        "annotated_image": image_to_data_url(annotated),
        "note": "Uses full-image and overlapping tile scans to improve small and aerial vehicle detection.",
    })


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 15 MB."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
