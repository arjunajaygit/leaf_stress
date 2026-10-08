"""
app.py (v2)
-----------
Advanced Crop Water Stress & Foliar Pathology Diagnostic Web System.
Features:
  - Multi-model inference (SVM RBF, Random Forest, Gradient Boosting, Logistic Regression, KNN)
  - Interactive Computer Vision Pipeline (Segmented Leaf, Pathology Decomposition, GLI Heatmap)
  - Explainable AI (XAI) feature attribution with biophysical anomaly scores
  - Crop Water Stress Index (CWSI 0-100%) and Agronomic Action Advisory
  - Comprehensive Model Benchmarking dashboard with 5-Fold Cross-Validation metrics
"""

import json
import tempfile
import warnings
from pathlib import Path
from uuid import uuid4

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template, jsonify
from werkzeug.exceptions import HTTPException

from config import (
    MODEL_PATH, BENCHMARK_PATH, MAX_CONTENT_LENGTH, MAX_IMAGE_PIXELS,
    ALLOWED_EXTENSIONS, server_options,
)
from feature_extraction import analyze_leaf_full, compute_cwsi, generate_agronomic_advisory

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH, MAX_IMAGE_PIXELS=MAX_IMAGE_PIXELS)

bundle = None
primary_model = None
all_models = {}
feature_names = []
classes = []
feature_stats = {}
benchmark_data = {}
model_load_error = None


def load_model_bundle(path=MODEL_PATH):
    """Load a trusted local artifact; leave the service unavailable on failure."""
    global bundle, primary_model, all_models, feature_names, classes, feature_stats, model_load_error
    bundle, primary_model, all_models = None, None, {}
    feature_names, classes, feature_stats = [], [], {}
    try:
        loaded = joblib.load(path)
        candidate = loaded["primary_model"] if isinstance(loaded, dict) else loaded
        names = list(loaded["feature_names_in_"] if isinstance(loaded, dict) else candidate.feature_names_in_)
        labels = list(loaded["classes_"] if isinstance(loaded, dict) else candidate.classes_)
        if not callable(getattr(candidate, "predict", None)) or not names or not labels:
            raise ValueError("Artifact has no usable classifier or feature/class schema")
        if len(set(names)) != len(names) or not all(isinstance(n, str) for n in names):
            raise ValueError("Artifact feature names are invalid")
        if list(getattr(candidate, "feature_names_in_", names)) != names:
            raise ValueError("Classifier and bundle feature order disagree")
        if list(getattr(candidate, "classes_", labels)) != labels:
            raise ValueError("Classifier and bundle class order disagree")
        bundle = loaded if isinstance(loaded, dict) else None
        primary_model, feature_names, classes = candidate, names, labels
        all_models = bundle.get("all_models", {"Primary": candidate}) if bundle else {"Default Model": candidate}
        feature_stats = bundle.get("feature_stats", {}) if bundle else {}
        model_load_error = None
        return True
    except Exception as exc:
        model_load_error = f"Cannot load model at {path}: {exc}"
        app.logger.error("Startup model error: %s", model_load_error)
        return False


load_model_bundle()
try:
    with open(BENCHMARK_PATH, encoding="utf-8") as stream:
        benchmark_data = json.load(stream)
except Exception:
    app.logger.exception("Startup benchmark error: cannot load %s", BENCHMARK_PATH)


class APIError(Exception):
    def __init__(self, message, status, code):
        super().__init__(message)
        self.status = status
        self.code = code


@app.errorhandler(APIError)
def api_error(exc):
    return jsonify(error=str(exc), code=exc.code), exc.status


@app.errorhandler(HTTPException)
def http_error(exc):
    response = exc.get_response()
    message = "Image upload exceeds the permitted request size." if exc.code == 413 else exc.description
    response.set_data(json.dumps({"error": message, "code": f"http_{exc.code}"}))
    response.content_type = "application/json"
    return response


@app.errorhandler(Exception)
def unexpected_error(exc):
    app.logger.exception("Unexpected request failure")
    return jsonify(error="An unexpected server error occurred. Please try again.", code="internal_error"), 500


def validate_image(path):
    """Check header dimensions before full OpenCV decoding, then check decoded pixels."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise APIError("Use a JPEG, PNG or WebP image.", 415, "unsupported_image")
                width, height = image.size
                if width * height > app.config["MAX_IMAGE_PIXELS"]:
                    raise APIError("Image dimensions are too large. Please upload a smaller image.", 413, "image_too_large")
                image.verify()
        decoded = cv2.imread(str(path))
        if decoded is None or decoded.size == 0:
            raise APIError("The file cannot be decoded as an image. Please choose another photo.", 400, "invalid_image")
        if decoded.shape[0] * decoded.shape[1] > app.config["MAX_IMAGE_PIXELS"]:
            raise APIError("Image dimensions are too large. Please upload a smaller image.", 413, "image_too_large")
    except APIError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise APIError("Image dimensions exceed the safe decoding limit.", 413, "image_too_large") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, cv2.error):
        raise APIError("The file is not a readable image. Please re-upload a JPEG, PNG or WebP photo.", 400, "invalid_image") from None


def compute_xai_attributions(features):
    """
    Computes Explainable AI feature anomaly scores relative to the training baseline.
    Highlights specific morphological and spectral traits that influenced the model.
    """
    if not feature_stats:
        return []

    feature_descriptions = {
        "h_std": "Hue Dispersion (Foliar color variance & spot irregularity)",
        "v_std": "Brightness Contrast (Dark necrotic lesions vs light canopy)",
        "extent": "Contour Extent (Bounding rectangle fill ratio)",
        "h_mean": "Mean Hue Angle (Chlorophyll green vs chlorosis yellow/brown)",
        "s_mean": "Color Saturation (Vividness of foliage pigmentation)",
        "s_std": "Saturation Variance (Spotting and irregular drying)",
        "b_mean": "Blue Channel Reflection (Pathology reflectance)",
        "gli_mean": "Green Leaf Index (Photosynthetic vegetative vitality)",
        "compactness": "Outline Compactness (Leaf curling and edge involution)",
        "solidity": "Convex Hull Solidity (Edge margin wilting & concavity)",
        "aspect_ratio": "Leaf Aspect Ratio (Width-to-length elongation)",
        "exg_mean": "Excess Green Index (Green vegetation contrast)",
        "r_mean": "Red Channel Reflectance (Carotenoid / anthocyanin expression)",
        "g_mean": "Green Channel Reflectance (Chlorophyll content)",
        "v_mean": "Visual Brightness (Surface reflection)",
    }

    attributions = []
    importances = bundle.get("feature_importance", {}) if bundle else {}

    for col in feature_names:
        val = features.get(col, 0.0)
        baseline = feature_stats.get(col, {"mean": val, "std": 1.0})
        mean = baseline["mean"]
        std = baseline["std"] if baseline["std"] > 1e-5 else 1.0

        z_score = (val - mean) / std
        weight = importances.get(col, 1.0 / len(feature_names))
        impact_score = abs(z_score) * weight

        # Interpret biological meaning of anomaly
        if col in ("solidity", "compactness") and z_score < -0.5:
            interpretation = f"Leaf edge curling and wilting detected ({abs(z_score):.1f}σ below normal)"
        elif col in ("gli_mean", "exg_mean") and z_score < -0.5:
            interpretation = f"Photosynthetic index degradation ({abs(z_score):.1f}σ below healthy baseline)"
        elif col in ("h_std", "v_std") and z_score > 0.5:
            interpretation = f"High color irregularity indicative of localized lesion/spotting (+{z_score:.1f}σ)"
        elif col == "h_mean" and z_score < -0.5:
            interpretation = f"Shift from green towards chlorotic yellow/amber hue ({abs(z_score):.1f}σ)"
        else:
            interpretation = f"Trait measurement: {val:.2f} (Baseline: {mean:.2f})"

        attributions.append({
            "feature": col,
            "label": feature_descriptions.get(col, col),
            "value": round(float(val), 3),
            "baseline_mean": round(float(mean), 3),
            "z_score": round(float(z_score), 2),
            "impact_score": round(float(impact_score), 4),
            "interpretation": interpretation,
        })

    # Sort by impact score descending
    attributions.sort(key=lambda x: x["impact_score"], reverse=True)
    return attributions


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/benchmark")
def get_benchmark():
    """Provides model comparison metrics and 5-fold cross-validation results."""
    return jsonify(benchmark_data)


@app.route("/predict", methods=["POST"])
def predict():
    if primary_model is None:
        raise APIError("The model is unavailable. Ask the operator to check the startup model error.", 503, "model_unavailable")
    if "image" not in request.files:
        raise APIError("No image uploaded. Please select a photo.", 400, "missing_image")
    file = request.files["image"]
    if not file.filename:
        raise APIError("No file selected. Please choose a photo.", 400, "missing_filename")
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise APIError("Use a .jpg, .jpeg, .png or .webp file.", 415, "unsupported_extension")

    # Client filenames never determine a filesystem path. Cleanup also runs on errors.
    with tempfile.TemporaryDirectory(prefix="lecor-upload-") as directory:
        save_path = Path(directory) / (uuid4().hex + extension)
        file.save(save_path)
        validate_image(save_path)
        try:
            features, visualizations, tissue_stats = analyze_leaf_full(str(save_path))
        except Exception:
            app.logger.exception("Image processing failed")
            raise APIError("Image processing failed. Please try another photo.", 500, "processing_error") from None

    if features is None:
        raise APIError("No leaf could be segmented. Please retake the photo against a contrasting background.", 422, "no_leaf")
    try:
        if set(features) != set(feature_names):
            raise ValueError("Extracted features do not match the model schema")
        X = pd.DataFrame([features], columns=feature_names)
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Features contain non-finite values")
    except Exception:
        app.logger.exception("Feature schema validation failed")
        raise APIError("The image features do not match the loaded model. Contact the operator.", 503, "schema_error") from None

    try:
        predictions = primary_model.predict(X)
        if np.asarray(predictions).shape != (1,) or predictions[0] not in classes:
            raise ValueError("Invalid prediction shape or class")
        pred = str(predictions[0])
        if hasattr(primary_model, "predict_proba"):
            probabilities = np.asarray(primary_model.predict_proba(X), dtype=float)
            model_classes = list(getattr(primary_model, "classes_", classes))
            if model_classes != list(classes) or probabilities.shape != (1, len(classes)):
                raise ValueError("Probability schema does not match classes")
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.isclose(probabilities.sum(), 1):
                raise ValueError("Invalid model probabilities")
            probs = {cls: round(float(p), 3) for cls, p in zip(model_classes, probabilities[0])}
        else:
            probs = {cls: 1.0 if cls == pred else 0.0 for cls in classes}
        comparison_predictions = {name: str(model.predict(X)[0]) for name, model in all_models.items()}
    except Exception:
        app.logger.exception("Model prediction failed")
        raise APIError("The loaded model could not produce a valid prediction. Contact the operator.", 503, "model_error") from None

    # Crop Water Stress Index (CWSI) and Agronomic Prescriptions
    cwsi = compute_cwsi(features, pred, probs)
    advisory = generate_agronomic_advisory(pred, cwsi, tissue_stats)
    xai_attributions = compute_xai_attributions(features)

    return jsonify({
        "prediction": pred,
        "probabilities": probs,
        "best_model_name": bundle.get("best_model_name", "Primary Model") if bundle else "Primary Model",
        "multi_model_predictions": comparison_predictions,
        "cwsi": cwsi,
        "tissue_stats": tissue_stats,
        "visualizations": visualizations,
        "advisory": advisory,
        "xai_attributions": xai_attributions[:6],  # Top 6 most influential biophysical features
        "features": {k: round(v, 3) for k, v in features.items()},
    })


if __name__ == "__main__":
    if model_load_error:
        raise SystemExit("Startup error: " + model_load_error)
    app.run(**server_options())
