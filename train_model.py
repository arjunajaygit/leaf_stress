"""
train_model.py (v2)
-------------------
Trains and benchmarks multiple machine learning architectures on leaf_features.csv
using Stratified 5-Fold Cross-Validation:
  1. Random Forest Classifier (Bagged Decision Ensembles)
  2. Support Vector Machine (RBF Kernel + StandardScaler)
  3. Gradient Boosting Classifier (Sequential Boosting)
  4. Logistic Regression (L2 Regularized Linear Classifier)
  5. K-Nearest Neighbors (Metric Space Classifier)

Exports:
  - leaf_stress_model_v1_<run>.pkl: Model bundle with best estimator, all trained models,
    feature baselines, and benchmark metadata.
  - model_benchmark_v1_<run>.json: Cross-validation comparison metrics, confusion matrices,
    and feature importance rankings for academic reporting and UI visualization.
"""

from config import CSV_PATH, versioned_path

import json
import hashlib
import config
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def run_benchmarks():
    print("==================================================")
    print("🌿 LEAF STRESS CLASSIFICATION - MULTI-MODEL BENCHMARK")
    print("==================================================")

    metadata_path = CSV_PATH.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise ValueError("Feature CSV has no preprocessing metadata. Build a new versioned dataset first; the original CSV is preserved.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("preprocessing") != config.preprocessing_signature() or metadata.get("csv_sha256") != hashlib.sha256(CSV_PATH.read_bytes()).hexdigest():
        raise ValueError("Feature CSV preprocessing metadata or content hash does not match. Rebuild the dataset.")
    df = pd.read_csv(CSV_PATH)
    required_classes = {"Healthy", "Moderate_Water_Stress", "Severe_Water_Stress", "Disease_Confounded"}
    if set(df.get("label", [])) != required_classes:
        raise ValueError("Quality-filtered dataset lacks one or more required classes. Collect acceptable images before four-class training.")
    if df["label"].value_counts().min() < 2:
        raise ValueError("At least two accepted samples per class are required even for provisional image-level CV.")
    feature_cols = [c for c in df.columns if c not in ("label", "filename")]
    X = df[feature_cols]
    y = df["label"]

    classes = sorted(y.unique().tolist())
    class_counts = y.value_counts().to_dict()
    print(f"\nTotal Dataset Samples: {len(df)}")
    print(f"Features Analyzed: {len(feature_cols)}")
    print(f"Class Distribution: {class_counts}\n")

    # Pooled dataset statistics for descriptive feature comparisons
    feature_stats = {}
    for col in feature_cols:
        feature_stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()) if float(df[col].std()) > 1e-6 else 1.0,
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    # Per-class baseline means (useful for reference profiles)
    class_profiles = {}
    for cls in classes:
        cls_df = df[df["label"] == cls]
        class_profiles[cls] = {col: float(cls_df[col].mean()) for col in feature_cols}

    # Models to benchmark
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_split=3, random_state=42
        ),
        "Support Vector Machine (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=2.5, probability=True, random_state=42)),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42
        ),
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.5, random_state=42)),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
    }

    n_splits = min(5, y.value_counts().min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    benchmark_results = {}
    trained_models = {}

    print(f"Running Stratified {n_splits}-Fold Cross-Validation across {len(models)} algorithms...\n")

    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv)

        acc = float(accuracy_score(y, y_pred))
        macro_p = float(precision_score(y, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y, y_pred, labels=classes).tolist()
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        benchmark_results[name] = {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
        }

        # Fit model on entire dataset for production bundle
        model.fit(X, y)
        trained_models[name] = model

        print(f"-> {name:30s} | Acc: {acc:.3f} | Macro F1: {macro_f1:.3f} | Weighted F1: {weighted_f1:.3f}")

    # Extract feature importance from Random Forest
    rf_model = trained_models["Random Forest"]
    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_cols, rf_model.feature_importances_)
    }
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("\nTop 8 Random Forest Feature Importances (development data) (Random Forest MDI):")
    for col, imp in list(sorted_importances.items())[:8]:
        print(f"   - {col:15s}: {imp:.4f}")

    # Select best model based on macro_f1
    best_model_name = max(benchmark_results.keys(), key=lambda k: benchmark_results[k]["macro_f1"])
    print(f"\nBest Performing Model: {best_model_name} (Macro F1 = {benchmark_results[best_model_name]['macro_f1']})")

    # Save benchmark metrics to JSON for web UI and reports
    benchmark_payload = {
        "dataset_summary": {
            "total_samples": len(df),
            "num_features": len(feature_cols),
            "classes": classes,
            "class_distribution": class_counts,
        },
        "models": benchmark_results,
        "feature_importance": sorted_importances,
        "feature_stats": feature_stats,
        "class_profiles": class_profiles,
        "best_model": best_model_name,
    }

    benchmark_path = versioned_path("model_benchmark", ".json")
    model_path = benchmark_path.with_name(benchmark_path.name.replace("model_benchmark", "leaf_stress_model")).with_suffix(".pkl")
    with open(benchmark_path, "x") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"Saved benchmark report to {benchmark_path}")

    # Save a new versioned bundle; preserve the original artifact.
    # Bundle contains fitted estimators and descriptive metadata.
    bundle = {
        "preprocessing": metadata["preprocessing"],
        "mask_quality": metadata["mask_quality"],
        "best_model_name": best_model_name,
        "primary_model": trained_models[best_model_name],
        "all_models": trained_models,
        "feature_names_in_": feature_cols,
        "classes_": classes,
        "feature_stats": feature_stats,
        "class_profiles": class_profiles,
        "feature_importance": sorted_importances,
        "benchmark": benchmark_results,
    }

    with open(model_path, "xb") as stream:
        joblib.dump(bundle, stream)
    print(f"Saved complete model and dataset-statistics bundle to {model_path}\n")


if __name__ == "__main__":
    run_benchmarks()
