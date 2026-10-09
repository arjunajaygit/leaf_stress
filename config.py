"""Filesystem and runtime settings shared by the application and scripts."""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
PREVIEW_ONLY = os.environ.get("LECOR_PREVIEW_ONLY", "false").lower() in {"1", "true", "yes"}


def project_path(variable, default):
    path = Path(os.environ.get(variable, default)).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


MODEL_PATH = project_path("LECOR_MODEL_PATH", "leaf_stress_model.pkl")
BENCHMARK_PATH = project_path("LECOR_BENCHMARK_PATH", "model_benchmark.json")
CSV_PATH = project_path("LECOR_CSV_PATH", "leaf_features.csv")
DATASET_DIR = BASE_DIR / "dataset"
MAX_CONTENT_LENGTH = int(os.environ.get("LECOR_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("LECOR_MAX_IMAGE_PIXELS", 16_000_000))
if MAX_CONTENT_LENGTH <= 0 or MAX_IMAGE_PIXELS <= 0:
    raise ValueError("LECOR upload and pixel limits must be positive integers.")
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def versioned_path(stem, suffix, run_id=None):
    """New artifacts never target the original filenames."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex
    return BASE_DIR / f"{stem}_v1_{run_id}{suffix}"


def server_options():
    return {
        "host": os.environ.get("LECOR_HOST", "127.0.0.1"),
        "port": int(os.environ.get("PORT", "5001")),
        "debug": os.environ.get("LECOR_DEBUG", "false").lower() in {"1", "true", "yes"},
    }

# Provisional acquisition/segmentation settings, not learned clinical thresholds.
WORKING_LONGEST_SIDE = 800
MORPH_REFERENCE_SIDE = 800
MORPH_KERNEL_AT_REFERENCE = 5
MASK_MIN_FOREGROUND_FRACTION = 0.02
MASK_MAX_FOREGROUND_FRACTION = 0.95
MASK_MIN_CONTOUR_AREA_FRACTION = 0.005
MASK_MIN_CONTOUR_AREA_PIXELS = 128
MASK_LARGE_COMPONENT_MIN_FRACTION = 0.01
MASK_LARGE_COMPONENT_RELATIVE_FRACTION = 0.20
MASK_MAX_LARGE_COMPONENTS = 1
MASK_REJECT_BORDER_CONTACT = True
HSV_LOWER = (10, 20, 20)
HSV_UPPER = (100, 255, 255)
MIN_CLASS_PROBABILITY = float(os.environ.get("LECOR_MIN_CLASS_PROBABILITY", "0.60"))
if not 0 < MIN_CLASS_PROBABILITY <= 1:
    raise ValueError("LECOR_MIN_CLASS_PROBABILITY must be in (0, 1].")


def preprocessing_signature():
    """Prevent new resized features being silently supplied to legacy models."""
    return {
        "version": "working-resolution-v1",
        "working_longest_side": WORKING_LONGEST_SIDE,
        "resize": "area-down-linear-up",
        "morph_reference_side": MORPH_REFERENCE_SIDE,
        "morph_kernel_at_reference": MORPH_KERNEL_AT_REFERENCE,
        "hsv_lower": list(HSV_LOWER), "hsv_upper": list(HSV_UPPER),
    }


def mask_quality_settings():
    return {
        "min_foreground_fraction": MASK_MIN_FOREGROUND_FRACTION,
        "max_foreground_fraction": MASK_MAX_FOREGROUND_FRACTION,
        "min_contour_area_fraction": MASK_MIN_CONTOUR_AREA_FRACTION,
        "min_contour_area_pixels": MASK_MIN_CONTOUR_AREA_PIXELS,
        "large_component_min_fraction": MASK_LARGE_COMPONENT_MIN_FRACTION,
        "large_component_relative_fraction": MASK_LARGE_COMPONENT_RELATIVE_FRACTION,
        "max_large_components": MASK_MAX_LARGE_COMPONENTS,
        "reject_border_contact": MASK_REJECT_BORDER_CONTACT,
    }
