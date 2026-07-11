import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def read_image(uploaded):
    if uploaded is None or uploaded.filename == "":
        raise ValueError("Both a clear-road reference and a current image are required.")
    if uploaded.mimetype not in ALLOWED_TYPES:
        raise ValueError("Use JPG, PNG, or WEBP images.")
    try:
        return Image.open(uploaded.stream).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("One of the files is not a readable image.") from exc


def detect_obstructions(reference: Image.Image, current: Image.Image, sensitivity: int, min_area_percent: float):
    current = current.resize(reference.size, Image.Resampling.LANCZOS)
    ref = cv2.cvtColor(np.array(reference), cv2.COLOR_RGB2BGR)
    cur = cv2.cvtColor(np.array(current), cv2.COLOR_RGB2BGR)

    ref_blur = cv2.GaussianBlur(ref, (11, 11), 0)
    cur_blur = cv2.GaussianBlur(cur, (11, 11), 0)
    diff = cv2.absdiff(ref_blur, cur_blur)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    image_area = reference.width * reference.height
    minimum_area = image_area * min_area_percent / 100.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    output = cur.copy()
    changed_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
        changed_area += area
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), max(2, reference.width // 350))
        cv2.putText(output, "OBSTRUCTION", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    overlay = output.copy()
    overlay[mask > 0] = (0, 70, 255)
    output = cv2.addWeighted(output, 0.78, overlay, 0.22, 0)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    changed_percent = round((changed_area / image_area) * 100, 2)
    return Image.fromarray(output_rgb), boxes, changed_percent


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/detect")
def detect():
    try:
        reference = read_image(request.files.get("reference"))
        current = read_image(request.files.get("current"))
        sensitivity = int(request.form.get("sensitivity", "35"))
        sensitivity = min(max(sensitivity, 10), 100)
        min_area = float(request.form.get("min_area", "0.35"))
        min_area = min(max(min_area, 0.05), 10.0)
        annotated, boxes, changed_percent = detect_obstructions(reference, current, sensitivity, min_area)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Obstruction detection failed")
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    return jsonify({
        "count": len(boxes),
        "status": "BLOCKED" if boxes else "CLEAR",
        "changed_percent": changed_percent,
        "boxes": boxes,
        "annotated_image": image_to_data_url(annotated),
        "note": "Red regions are large visual changes compared with the clear-road reference. Best results require the same fixed camera position.",
    })


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Combined upload is too large. Maximum size is 30 MB."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
