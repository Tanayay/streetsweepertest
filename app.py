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


def build_interference_mask(bgr: np.ndarray, sensitivity: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.12):, :] = 255

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    saturation = hsv[:, :, 1]
    color_mask = cv2.inRange(saturation, max(18, 82 - sensitivity), 255)

    local_average = cv2.GaussianBlur(blurred, (0, 0), 18)
    local_difference = cv2.absdiff(blurred, local_average)
    _, contrast_mask = cv2.threshold(
        local_difference,
        max(7, 34 - sensitivity // 3),
        255,
        cv2.THRESH_BINARY,
    )

    edges = cv2.Canny(
        blurred,
        max(18, 88 - sensitivity),
        max(55, 185 - sensitivity),
    )
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = cv2.bitwise_or(color_mask, contrast_mask)
    mask = cv2.bitwise_or(mask, edges)
    mask = cv2.bitwise_and(mask, roi_mask)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, medium_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    mask = cv2.dilate(mask, small_kernel, iterations=1)
    return mask


def watershed_regions(bgr: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return []

    # Peaks inside each obstruction become seeds. Watershed then splits touching cars.
    normalized = distance / distance.max()
    sure_foreground = np.uint8(normalized > 0.24) * 255
    sure_foreground = cv2.morphologyEx(
        sure_foreground,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    sure_background = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr.copy(), markers.astype(np.int32))

    regions = []
    for marker_id in range(2, int(markers.max()) + 1):
        region = np.uint8(markers == marker_id) * 255
        if cv2.countNonZero(region) > 0:
            regions.append(region)
    return regions


def car_like_boxes(
    bgr: np.ndarray,
    mask: np.ndarray,
    min_area_percent: float,
) -> tuple[list[list[int]], float]:
    h, w = mask.shape
    image_area = w * h
    minimum_area = image_area * min_area_percent / 100.0
    maximum_area = image_area * 0.24

    regions = watershed_regions(bgr, mask)
    if not regions:
        regions = [mask]

    boxes = []
    total_area = 0.0

    for region in regions:
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum_area or area > maximum_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < w * 0.025 or bh < h * 0.025:
                continue

            aspect = max(bw, bh) / max(1, min(bw, bh))
            rectangularity = area / max(1, bw * bh)
            hull = cv2.convexHull(contour)
            solidity = area / max(1.0, cv2.contourArea(hull))

            # Broad car-shape filters. Perspective can make cars nearly square,
            # so the limits intentionally remain forgiving.
            if aspect > 5.5:
                continue
            if rectangularity < 0.22 or solidity < 0.38:
                continue

            padding_x = max(3, int(bw * 0.04))
            padding_y = max(3, int(bh * 0.04))
            boxes.append([
                max(0, x - padding_x),
                max(0, y - padding_y),
                min(w - 1, x + bw + padding_x),
                min(h - 1, y + bh + padding_y),
            ])
            total_area += area

    # Remove nearly identical overlapping boxes caused by nearby watershed seeds.
    kept = []
    for box in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        x1, y1, x2, y2 = box
        area_a = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for existing in kept:
            ex1, ey1, ex2, ey2 = existing
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_b = max(1, (ex2 - ex1) * (ey2 - ey1))
            union = area_a + area_b - intersection
            if intersection / union > 0.55:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    covered_percent = round((total_area / image_area) * 100, 2)
    return kept, covered_percent


def detect_individual_cars(
    image: Image.Image,
    sensitivity: int,
    min_area_percent: float,
):
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = build_interference_mask(bgr, sensitivity)
    boxes, covered_percent = car_like_boxes(bgr, mask, min_area_percent)

    output = bgr.copy()
    overlay = output.copy()
    overlay[mask > 0] = (0, 70, 255)
    output = cv2.addWeighted(output, 0.88, overlay, 0.12, 0)

    line_width = max(2, image.width // 350)
    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), line_width)
        cv2.putText(
            output,
            f"CAR {index}",
            (x1, max(25, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
        )

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
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
        min_area = min(max(float(request.form.get("min_area", "0.25")), 0.03), 5.0)
        annotated, boxes, covered_percent = detect_individual_cars(
            image,
            sensitivity,
            min_area,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Individual car detection failed")
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    return jsonify({
        "count": len(boxes),
        "status": "CARS FOUND" if boxes else "NO CARS FOUND",
        "covered_percent": covered_percent,
        "boxes": boxes,
        "annotated_image": image_to_data_url(annotated),
        "note": "Experimental single-image mode: watershed separation splits connected obstruction regions into individual car-sized candidates. Shadows, sidewalks, trees, and unusual pavement can still cause false detections.",
    })


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 20 MB."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
