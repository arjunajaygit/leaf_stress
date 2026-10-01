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
import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

from feature_extraction import (
    analyze_leaf_full,
    compute_cwsi,
    generate_agronomic_advisory,
)

app = Flask(__name__)

MODEL_PATH = "leaf_stress_model.pkl"
BENCHMARK_PATH = "model_benchmark.json"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model bundle and benchmark data
bundle = None
primary_model = None
all_models = {}
feature_names = []
classes = []
feature_stats = {}
benchmark_data = {}

if os.path.exists(MODEL_PATH):
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict) and "primary_model" in loaded:
        bundle = loaded
        primary_model = bundle["primary_model"]
        all_models = bundle.get("all_models", {"Primary": primary_model})
        feature_names = bundle["feature_names_in_"]
        classes = bundle["classes_"]
        feature_stats = bundle.get("feature_stats", {})
    else:
        # Fallback for raw scikit-learn model
        primary_model = loaded
        feature_names = list(getattr(loaded, "feature_names_in_", []))
        classes = list(getattr(loaded, "classes_", []))
        all_models = {"Default Model": primary_model}

if os.path.exists(BENCHMARK_PATH):
    try:
        with open(BENCHMARK_PATH, "r") as f:
            benchmark_data = json.load(f)
    except Exception as e:
        print(f"Warning: could not load benchmark JSON: {e}")


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
        return jsonify({"error": "Model bundle not loaded. Run train_model.py first."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Selected file is empty"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # Full computer vision pipeline extraction
    features, visualizations, tissue_stats = analyze_leaf_full(save_path)
    if features is None:
        return jsonify({
            "error": "Could not detect or segment a leaf in this image. Please ensure the leaf is clearly visible against a contrasting background."
        }), 400

    # Align features with training schema
    X = pd.DataFrame([features])[feature_names]

    # Primary model prediction
    pred = primary_model.predict(X)[0]
    if hasattr(primary_model, "predict_proba"):
        proba = primary_model.predict_proba(X)[0]
        model_classes = list(getattr(primary_model, "classes_", classes))
        probs = {cls: round(float(p), 3) for cls, p in zip(model_classes, proba)}
    else:
        probs = {cls: 1.0 if cls == pred else 0.0 for cls in classes}

    # Comparative multi-model predictions for academic demonstration
    comparison_predictions = {}
    for m_name, mdl in all_models.items():
        try:
            m_pred = mdl.predict(X)[0]
            comparison_predictions[m_name] = str(m_pred)
        except Exception:
            pass

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
    port = int(os.environ.get("PORT", 5001))
    print(f"🌿 Leaf Stress Diagnostic System v2 running on http://0.0.0.0:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
