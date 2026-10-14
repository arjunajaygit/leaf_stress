"""Interactive manual SAM point annotation and dataset review server.

Provides a fast, modern browser interface to step through dataset images,
click Keep and Exclude points, review the live SAM segmentation, and save
high-quality segmented images directly to the dataset.
"""

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading

import cv2
from flask import Flask, jsonify, render_template, request, send_file
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

import plant_features as pf

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset_wholeplant_v1_20261013_original"
if not DEFAULT_DATASET.exists():
    DEFAULT_DATASET = ROOT / "dataset_wholeplant_v1_20261013"
DEFAULT_OUTPUT = ROOT / "dataset_wholeplant_v1_20261013"
CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"
MANIFEST_PATH = ROOT / "dataset_sam_review_manifest.json"
CLASSES = ["Healthy", "Moderate_Water_Stress", "Severe_Water_Stress", "Disease_Confounded"]

app = Flask(__name__)
predictor = None
lock = threading.Lock()
config_state = {
    "dataset_dir": DEFAULT_DATASET,
    "output_dir": DEFAULT_OUTPUT,
}


def get_predictor():
    global predictor
    if predictor is None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found at {CHECKPOINT}")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        model = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
        model.to(device="cpu")
        model.eval()
        predictor = SamPredictor(model)
    return predictor


def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


@app.route("/")
def index():
    return render_template("sam_annotator.html")


@app.route("/api/images")
def get_images():
    manifest = load_manifest()
    dataset_dir = config_state["dataset_dir"]
    images = []

    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                rel = f"{cls}/{p.name}"
                status = manifest.get(rel, {}).get("status", "pending")
                images.append({
                    "filename": rel,
                    "name": p.name,
                    "class": cls,
                    "status": status,
                })
    return jsonify(images)


@app.route("/api/image/<path:filename>")
def serve_image(filename):
    path = config_state["dataset_dir"] / filename
    if not path.is_file():
        return jsonify(error="Image not found"), 404
    return send_file(path)


@app.route("/api/suggest-points")
def suggest_points():
    filename = request.args.get("filename")
    if not filename:
        return jsonify(error="Missing filename"), 400

    path = config_state["dataset_dir"] / filename
    image = cv2.imread(str(path))
    if image is None:
        return jsonify(error="Could not decode image"), 400

    h, w = image.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cand = cv2.inRange(hsv, np.array(pf.SETTINGS["hsv_lower"]), np.array(pf.SETTINGS["hsv_upper"]))
    b, g, r = cv2.split(image.astype(float))
    cand[(g < pf.SETTINGS["green_red_ratio_min"] * r) | (g < pf.SETTINGS["green_blue_ratio_min"] * b)] = 0

    points = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    sorted_idx = np.argsort(areas)[::-1] if len(areas) else []

    for idx in sorted_idx[:3]:
        if areas[idx] > 20:
            cx, cy = centroids[idx + 1]
            points.append({"x": round(float(cx) / (w - 1), 4), "y": round(float(cy) / (h - 1), 4), "label": 1})

    if not points:
        points.append({"x": 0.5, "y": 0.5, "label": 1})

    # Add 1 exclude point if background exists
    dist_bg = cv2.distanceTransform(255 - cand, cv2.DIST_L2, 5)
    _, _, _, max_bg_loc = cv2.minMaxLoc(dist_bg)
    points.append({"x": round(float(max_bg_loc[0]) / (w - 1), 4), "y": round(float(max_bg_loc[1]) / (h - 1), 4), "label": 0})

    return jsonify(points=points)


@app.route("/api/segment", methods=["POST"])
def segment():
    data = request.get_json() or {}
    filename = data.get("filename")
    points = data.get("points", [])

    if not filename or not points:
        return jsonify(error="Missing filename or points"), 400

    path = config_state["dataset_dir"] / filename
    image = cv2.imread(str(path))
    if image is None:
        return jsonify(error="Could not decode image"), 400

    orig_h, orig_w = image.shape[:2]
    scale = min(1.0, 800 / max(orig_h, orig_w))
    if scale < 1.0:
        resized = cv2.resize(image, (round(orig_w * scale), round(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
    h, w = resized.shape[:2]

    coords = np.array([[p["x"] * (w - 1), p["y"] * (h - 1)] for p in points], dtype=np.float32)
    labels = np.array([p["label"] for p in points], dtype=np.int32)

    with lock:
        p_model = get_predictor()
        p_model.set_image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        masks, scores, _ = p_model.predict(point_coords=coords, point_labels=labels, multimask_output=False)
        p_model.reset_image()

    mask = masks[0]
    isolated = resized.copy()
    isolated[~mask] = 220

    overlay = resized.copy()
    overlay[mask] = (resized[mask] * 0.55 + np.array([30, 210, 30]) * 0.45).astype(np.uint8)

    def to_b64(img):
        ok, enc = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(enc).decode("ascii")

    return jsonify({
        "overlay": to_b64(overlay),
        "image": to_b64(isolated),
        "mask": to_b64(np.uint8(mask) * 255),
        "score": float(scores[0]),
        "selected_fraction": float(mask.mean()),
        "width": w,
        "height": h,
    })


@app.route("/api/save", methods=["POST"])
def save():
    data = request.get_json() or {}
    filename = data.get("filename")
    points = data.get("points", [])
    decision = data.get("decision", "approved")

    if not filename:
        return jsonify(error="Missing filename"), 400

    path = config_state["dataset_dir"] / filename
    out_path = config_state["output_dir"] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(path))
    if image is None:
        return jsonify(error="Could not decode image"), 400

    h, w = image.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
    h, w = resized.shape[:2]

    coords = np.array([[p["x"] * (w - 1), p["y"] * (h - 1)] for p in points], dtype=np.float32)
    labels = np.array([p["label"] for p in points], dtype=np.int32)

    with lock:
        p_model = get_predictor()
        p_model.set_image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        masks, scores, _ = p_model.predict(point_coords=coords, point_labels=labels, multimask_output=False)
        p_model.reset_image()

    mask = masks[0]
    isolated = resized.copy()
    isolated[~mask] = 220

    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(out_path), isolated, [cv2.IMWRITE_JPEG_QUALITY, 98])
    else:
        cv2.imwrite(str(out_path), isolated)

    manifest = load_manifest()
    manifest[filename] = {
        "status": decision,
        "points": points,
        "score": float(scores[0]),
        "selected_fraction": float(mask.mean()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dimensions": [w, h],
    }
    save_manifest(manifest)

    return jsonify(ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Source images to review")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory to save reviewed images")
    parser.add_argument("--port", type=int, default=5005, help="Port to run review tool on")
    args = parser.parse_args()

    config_state["dataset_dir"] = args.dataset.resolve()
    config_state["output_dir"] = args.output.resolve()

    print(f"Starting SAM Review Tool...")
    print(f"  Source dataset: {config_state['dataset_dir']}")
    print(f"  Output dataset: {config_state['output_dir']}")
    print(f"  Open in browser: http://127.0.0.1:{args.port}\n")

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
