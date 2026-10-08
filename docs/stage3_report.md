# Stage 3 — Segmentation quality gate

Date: 2026-10-08 (Asia/Kolkata). Branch: `audit-fixes`.

## Changes

- Centralized provisional segmentation and confidence thresholds in `config.py`.
- Standardized training, inference and audit images to longest side 800 pixels with aspect ratio preserved. Smaller images are upsampled without claiming added information. The morphology kernel scales with resolution (5×5 at 800 pixels, odd-sized at other resolutions).
- Added a shared `prepare_leaf()` path for preprocessing, candidate segmentation, quality decisions and the existing 15-feature schema. Rejected inputs do not produce training feature rows or web classifications.
- Added checks for border contact, excessive/insufficient selected foreground fraction, multiple substantial candidate components and tiny contours. All reasons are retained. Components are counted before selecting the largest contour, so selection cannot hide a second substantial object.
- Added selected-mask and candidate-mask previews with readable retake messages. Rejected images receive HTTP 422 and no prediction/indicator.
- Added “Uncertain / unsupported input” when maximum class probability is below `LECOR_MIN_CLASS_PROBABILITY` (default 0.60). Comparison uses unrounded probabilities. The threshold is explicitly provisional and calibration is unverified. No class, indicator or other-model classifications are shown for this outcome.
- Added `report_mask_quality.py`, the requested `reports/mask_quality_report.csv`, and `reports/mask_quality_summary.md` with all failure counts and filenames. Each CSV row represents one byte-unique image; aliases are retained in a separate column. Report creation refuses existing paths, and reruns can use new output names.
- Updated README and regression tests for the changed preprocessing and API behavior.

## Compatibility safeguard required by the preprocessing change

Resizing and morphology change numerical feature values. Silently feeding these values into the original saved model would create training/inference mismatch. The app therefore requires an exact preprocessing signature before classifying a quality-passing input. The original model still loads for historical metadata, but cannot produce predictions with the new extractor; the UI displays “Model update required” with mask previews (HTTP 503).

New versioned CSVs receive a preprocessing/quality metadata sidecar and a CSV SHA-256. The trainer refuses old/mismatched input metadata, missing classes and fewer than two accepted images per class. New fitted bundles would carry the same preprocessing signature. This small compatibility check is necessary now; complete schema/library/training provenance remains Stage 5 work.

**No replacement production model has been fitted or selected.** The accepted data contains no Moderate or Severe images, so four-class retraining is blocked. Do not relabel rejected data, relax gates just to recover class counts, or reinterpret a two-class model as a four-class model.

## Dataset report

| Class | Accepted unique images | Rejected unique images |
|---|---:|---:|
| Healthy | 4 | 4 |
| Moderate Water Stress | 0 | 9 |
| Severe Water Stress | 0 | 11 |
| Disease Confounded | 7 | 5 |
| Total | 11 | 29 |

| Reason | Images failing this reason | Images with this primary reason |
|---|---:|---:|
| `border_contact` | 29 | 29 |
| `foreground_too_large` | 9 | 0 |
| `foreground_too_small` | 0 | 0 |
| `multiple_components` | 0 | 0 |
| `tiny_contour` | 0 | 0 |
| `no_leaf` | 0 | 0 |
| `unreadable_image` | 0 | 0 |

Counts overlap: the 9 excessive-foreground images are also among the 29 border-contact images. The summary includes a full configuration snapshot and rejected-file list. No multiple-component failures does not mean those images contain one leaf: connected foliage can merge into one component, which may instead fail the border/fraction checks.

Defaults: foreground fraction must be within 0.02–0.95 inclusive; contour area must reach max(128 pixels², 0.005 × image area); a large component must occupy at least 0.01 × image area and at least 0.20 × largest-component area; at most one such component is allowed. Any selected-mask border contact fails. All are provisional, not trained or physiologically validated thresholds. Natural stems at the border may trigger conservative rejection; review actual images and the single-leaf acquisition requirement.

## Actual verification

Commands:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node tests/frontend_stage1.cjs
git diff --check
```

- **65 Python tests passed** (7.93 seconds in the recorded run).
- Tests cover standardized dimensions, scaled kernels, each mask failure, overlapping reasons, multiple-component handling before selection, tiny specks, threshold boundaries, unreadable images, shared training/inference features and rejections, preview/no-class contracts, confidence rejection before rounding, configurable confidence threshold and explicit legacy-model refusal.
- All Stage 1 security/crash and Stage 2 claim regressions pass. Test-only compatible model fixtures exercise successful classification and uncertainty; they are not production artifacts or scientific evaluation results.
- Report regeneration in a temporary directory matches the committed CSV, and attempts to overwrite a report fail.
- Frontend logic tests pass rejected/uncertain/incompatible-model routing, both mask previews, all reason codes, resetting and previous lifecycle tests. Real-device/browser acceptance remains pending.
- Actual Flask sweep using the unchanged shipped model over all 40 unique images: **29 HTTP 422 rejected; 11 HTTP 503 model_update_required**. Every response omits a class and indicator. No runtime 500 errors occurred in that sweep.
- SHA-256 checks confirm all protected originals and backup copies remain unchanged. No dataset images were removed and the SRS PDF was not edited.
- 1,037 existing joblib/NumPy deprecation warnings remain in the Python run. No test failed.

## Limits and next steps

A centered green rectangular object can pass the mask gate; a test records this limitation. The gate evaluates geometry/acquisition, not semantic leaf identity or symptom cause. Model confidence cannot guarantee out-of-distribution rejection. Probability calibration is unverified.

The owner should review `reports/mask_quality_summary.md` and its rejected images, then provide appropriate single-leaf Moderate and Severe examples plus provenance/group identifiers when collecting data. A sufficiently representative four-class dataset and compatible retraining are required before normal classification can resume. New images must not be invented or labelled from color/shape alone.

Stage 4 manifest and grouped evaluation, Stage 5 ablations/reproducibility and Stage 6 expanded documentation/testing remain pending. No new accuracy or scientifically independent evaluation numbers are claimed. Stop after this stage and wait for “continue”.
