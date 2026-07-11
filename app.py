import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def read_image(uploaded):
    if uploaded is None or uploaded.filename == "":
        raise ValueError("Choose or paste a street image first.")
    if uploaded.mimetype not in ALLOWED_TYPES:
        raise ValueError("Use a JPG, PNG, or WEBP image.")
    try:
        return Image.open(uploaded.stream).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("That file is not a readable image.") from exc


def detect_interference(image: Image.Image, sensitivity: int, min_area_percent: float):
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Assume the useful road region is mostly in the lower 85% of the frame.
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.15):, :] = 255

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Road is usually low-saturation and relatively smooth. Objects tend to add
    # stronger color, edges, and local contrast.
    saturation = hsv[:, :, 1]
    color_mask = cv2.inRange(saturation, max(20, 85 - sensitivity), 255)

    local_average = cv2.GaussianBlur(blurred, (0, 0), 18)
    local_difference = cv2.absdiff(blurred, local_average)
    _, contrast_mask = cv2.threshold(
        local_difference,
        max(8, 35 - sensitivity // 3),
        255,
        cv2.THRESH_BINARY,
    )

    edges = cv2.Canny(blurred, max(20, 90 - sensitivity), max(60, 190 - sensitivity))
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    mask = cv2.bitwise_or(color_mask, contrast_mask)
    mask = cv2.bitwise_or(mask, edges)
    mask = cv2.bitwise_and(mask, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    image_area = w * h
    minimum_area = image_area * min_area_percent / 100.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = bgr.copy()
    boxes = []
    total_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)

        # Ignore very thin lines that are likely lane markings or image borders.
        if bw < w * 0.03 or bh < h * 0.03:
            continue

        boxes.append([x, y, x + bw, y + bh])
        total_area += area
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 0, 255), max(2, w // 350))
        cv2.putText(
            output,
            "INTERFERENCE",
            (x, max(26, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

    overlay = output.copy()
    overlay[mask > 0] = (0, 70, 255)
    output = cv2.addWeighted(output, 0.82, overlay, 0.18, 0)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    covered_percent = round((total_area / image_area) * 100, 2)
    return Image.fromarray(output_rgb), boxes, covered_percent


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/detect")
def detect():
    try:
        image = read_image(request.files.get("image"))
        sensitivity = min(max(int(request.form.get("sensitivity", "55")), 10), 90)
        min_area = min(max(float(request.form.get("min_area", "0.45")), 0.05), 10.0)
        annotated, boxes, covered_percent = detect_interference(image, sensitivity, min_area)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Road interference detection failed")
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    return jsonify({
        "count": len(boxes),
        "status": "BLOCKED" if boxes else "CLEAR",
        "covered_percent": covered_percent,
        "boxes": boxes,
        "annotated_image": image_to_data_url(annotated),
        "note": "Single-image experimental mode: red areas are large non-road-like regions. Shadows, markings, sidewalks, and unusual pavement can still create false detections.",
    })


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 20 MB."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
