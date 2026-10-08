# Stage 1 — Bugs and security

Date: 2026-10-08 (Asia/Kolkata)
Branch: `audit-fixes`

## Changes

- Copied `dataset/`, `leaf_features.csv`, `leaf_stress_model.pkl`, and `model_benchmark.json` to `backup_original/`, with SHA-256 verification. Original inputs and artifacts remain unchanged.
- Guarded empty chlorosis/necrosis selections before blending overlays.
- Uploads use UUID filenames with JPEG/PNG/WebP extensions in per-request temporary directories. Cleanup runs on success and failure. Existing `uploads/` files are untouched.
- Added a 10 MiB request limit and 16 million decoded-pixel limit. Pillow checks image format/dimensions before OpenCV decoding; OpenCV must then successfully decode the file. Pillow is now an explicit dependency.
- Added consistent JSON errors for invalid input, request limits, segmentation failure, processing failure, model/schema failure, HTTP errors and unexpected exceptions. Details are logged on the server rather than exposed to clients.
- Model load failure is logged at startup. Direct CLI startup exits with a clear error. When imported by a WSGI host, prediction returns 503 until a valid model is available.
- Added `config.py` for project-relative paths and runtime settings. Dataset generation and training now write uniquely named versioned artifacts using exclusive creation. Application defaults still load the original model/benchmark; no new production model was selected.
- Default server binding is `127.0.0.1`, with debug disabled.
- Frontend clears previous outputs at selection/start/failure, locks Reset and input during requests, revokes preview URLs, and handles non-JSON/malformed responses with readable messages.
- Added Python regression/security tests and an executable JavaScript test for the actual inline frontend logic.

## Configuration

| Variable | Default / meaning |
|---|---|
| `LECOR_HOST` | `127.0.0.1`; explicitly set another address for LAN access |
| `PORT` | `5001` |
| `LECOR_DEBUG` | `false`; enable only intentionally for local development |
| `LECOR_MAX_UPLOAD_BYTES` | `10485760`, including multipart request overhead |
| `LECOR_MAX_IMAGE_PIXELS` | `16000000` |
| `LECOR_MODEL_PATH` | `leaf_stress_model.pkl` |
| `LECOR_BENCHMARK_PATH` | `model_benchmark.json` |
| `LECOR_CSV_PATH` | `leaf_features.csv`; set this to a newly generated CSV to train from it |

Relative artifact paths resolve against the project directory. Model artifacts must come from a trusted source because joblib uses pickle. Generated filenames contain `_v1_` plus a timestamp and UUID; the model and benchmark from one training run share the same identifier. This filename version is not yet the schema version scheduled for Stage 5.

## Verification

Commands:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node tests/frontend_stage1.cjs
git diff --check
```

- Python: 31 tests passed, including both healthy-image regressions, upload cleanup, traversal-style client filenames, byte/pixel limits, JSON errors, model/schema failures, startup behavior, changed working directory, backup checks and versioned artifact generation in a temporary directory.
- JavaScript: passed stale-result clearing, in-flight Reset lock, object URL lifecycle, non-JSON and malformed JSON errors, server errors, successful completion and reset. This is a minimal-DOM logic test, not a real-device/browser acceptance test.
- Full extraction/visualization sweep: 40/40 unique dataset images completed successfully.
- Protected originals and backup copies matched the SHA-256 manifest.
- Existing dependency deprecation warnings occur when joblib loads the original model (NumPy array shape assignment). The temporary training test also exercises the existing SVM probability option, which emits a deprecation warning. These did not fail tests; dependency pinning and model metadata remain scheduled for Stage 5.

## Remaining work

Stage 2 claim corrections, Stage 3 mask rejection/uncertainty, Stage 4 provenance/grouped evaluation, Stage 5 feature experiments/reproducibility, and Stage 6 expanded tests/documentation remain pending. The current model, score, advisories and segmentation are not scientifically validated by these software tests. Camera, printing and cross-browser behavior still need manual device verification. No source labels or plant/session identities were invented.
