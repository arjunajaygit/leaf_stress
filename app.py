"""Flask interface for experimental visual leaf classification.

Uses a saved classifier, shared image features, heuristic visualizations and
comparisons with dataset averages. Outputs do not establish physiological status.
"""

import config
from model_contract import validate_bundle, explain_logistic
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
from feature_extraction import analyze_leaf_full, compute_visual_stress_indicator, generate_agronomic_advisory

if config.ANALYSIS_MODE == "wholeplant":
    from plant_features import analyze_plant_full as analyze_leaf_full
    from plant_features import relative_indicator as compute_visual_stress_indicator
    from plant_features import preprocessing_signature as active_preprocessing_signature
else:
    active_preprocessing_signature = config.preprocessing_signature

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH, MAX_IMAGE_PIXELS=MAX_IMAGE_PIXELS,
                  MIN_CLASS_PROBABILITY=config.MIN_CLASS_PROBABILITY, PREVIEW_ONLY=config.PREVIEW_ONLY)

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
        validate_bundle(loaded)
        if loaded.get("analysis_mode", "leaf") != config.ANALYSIS_MODE:
            raise ValueError("Model analysis mode differs from the selected application mode")
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


if not config.PREVIEW_ONLY:
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


def compare_feature_measurements(features):
    """Describe deviations from pooled dataset averages, not prediction contributions."""
    descriptions = {
        "h_mean": "Mean hue (OpenCV units)", "h_std": "Hue standard deviation",
        "s_mean": "Mean saturation", "s_std": "Saturation standard deviation",
        "v_mean": "Mean brightness", "v_std": "Brightness standard deviation",
        "r_mean": "Mean red channel", "g_mean": "Mean green channel",
        "b_mean": "Mean blue channel", "exg_mean": "Mean excess green index",
        "gli_mean": "Mean green leaf index", "solidity": "Projected shape solidity",
        "aspect_ratio": "Axis-aligned projected width/height ratio",
        "extent": "Projected contour bounding-box occupancy",
        "compactness": "Projected shape circularity",
    }
    comparisons = []
    for name in feature_names:
        baseline = feature_stats.get(name)
        if baseline is None:
            continue
        value = float(features[name])
        mean, std = float(baseline["mean"]), float(baseline["std"])
        z_score = (value - mean) / std if std > 1e-5 else None
        comparisons.append({
            "feature": name,
            "label": descriptions.get(name, name),
            "value": round(value, 3),
            "dataset_mean": round(mean, 3),
            "z_score": round(z_score, 2) if z_score is not None else None,
            "interpretation": "Compared with the pooled dataset across all classes; not a healthy reference or a contribution to the prediction.",
        })
    return comparisons


@app.route("/")
def index():
    return render_template("index.html", preview_only=app.config["PREVIEW_ONLY"], classifier_available=primary_model is not None, wholeplant=config.ANALYSIS_MODE == "wholeplant")


@app.route("/api/benchmark")
def get_benchmark():
    """Provides saved image-level development CV metrics, not an independent test estimate."""
    if app.config["PREVIEW_ONLY"]:
        return jsonify(available=False, message="Preview-only mode: historical classification numbers are not demonstration performance.")
    return jsonify(benchmark_data)


@app.route("/predict", methods=["POST"])
def predict():
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
            features, visualizations, tissue_stats, quality = analyze_leaf_full(str(save_path))
        except Exception:
            app.logger.exception("Image processing failed")
            raise APIError("Image processing failed. Please try another photo.", 500, "processing_error") from None

    if not quality["passed"]:
        return jsonify({
            "outcome": "rejected", "error": quality["message"], "code": quality["reason_code"],
            "prediction": None, "visual_stress_indicator": None,
            "quality": quality, "visualizations": visualizations,
        }), 422
    if app.config["PREVIEW_ONLY"]:
        return jsonify(outcome="preview_only", code="classification_disabled",
                       message="Mask preview only. Classification is unavailable until a compatible model is trained. No validated classifier exists yet.",
                       prediction=None, visual_stress_indicator=None, quality=quality, visualizations=visualizations)
    if primary_model is None or not bundle or bundle.get("preprocessing") != active_preprocessing_signature():
        return jsonify({
            "outcome": "model_update_required", "code": "model_unavailable" if primary_model is None else "preprocessing_mismatch",
            "error": "This image passed the provisional mask checks, but no compatible model is loaded. The operator must check the startup schema/version error and supply a compatible trained model.",
            "prediction": None, "visual_stress_indicator": None,
            "quality": quality, "visualizations": visualizations,
        }), 503
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
            raise ValueError("Class probabilities are required for provisional uncertainty rejection")
    except Exception:
        app.logger.exception("Model prediction failed")
        raise APIError("The loaded model could not produce a valid prediction. Contact the operator.", 503, "model_error") from None

    threshold = app.config["MIN_CLASS_PROBABILITY"]
    top_probability = float(probabilities.max())
    if top_probability < threshold:
        return jsonify({
            "outcome": "uncertain", "message": "Uncertain / unsupported input",
            "detail": "The highest model class probability is below the provisional threshold. Retake a clearer photo or seek independent assessment; a passing mask does not verify leaf identity.",
            "prediction": None, "visual_stress_indicator": None,
            "top_probability": top_probability, "probability_threshold": threshold,
            "threshold_provisional": True, "quality": quality, "visualizations": visualizations,
        })
    try:
        comparison_predictions = {name: str(model.predict(X)[0]) for name, model in all_models.items()
                                  if model is not primary_model}
    except Exception:
        app.logger.exception("Comparison model prediction failed")
        raise APIError("A comparison model failed. Contact the operator.", 503, "model_error") from None

    # Experimental heuristic and descriptive measurements; no physiological claims.
    indicator = compute_visual_stress_indicator(features, pred, probs)
    advisory = generate_agronomic_advisory(pred)
    feature_comparisons = compare_feature_measurements(features)

    return jsonify({
        "outcome": "classified",
        "quality": quality,
        "probability_threshold": threshold,
        "threshold_provisional": True,
        "prediction": pred,
        "probabilities": probs,
        "best_model_name": bundle.get("best_model_name", "Primary Model") if bundle else "Primary Model",
        "multi_model_predictions": comparison_predictions,
        "visual_stress_indicator": indicator,
        "tissue_stats": tissue_stats,
        "visualizations": visualizations,
        "advisory": advisory,
        "feature_comparisons": feature_comparisons,
        "model_explanation": explain_logistic(primary_model, X, pred),
        "features": {k: round(v, 3) for k, v in features.items()},
    })


def main():
    if model_load_error and not app.config["PREVIEW_ONLY"]:
        raise SystemExit("Startup error: " + model_load_error)
    app.run(**server_options())


if __name__ == "__main__":
    main()
