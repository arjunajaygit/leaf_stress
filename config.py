"""Filesystem and runtime settings shared by the application and scripts."""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent


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
