# Lecor — experimental whole-plant visual assessment

A working **OpenCV + scikit-learn + Flask** demonstration now accepts a whole tomato-plant photo, extracts candidate foliage across the image, and returns a dataset-category prediction or an uncertainty outcome. It does not identify every leaf or assess hidden leaves/roots.

**A classifier is trained and runnable, but no scientifically validated classifier exists yet.** The new dataset has 44 images (11 per folder label). Plant/session identities and label provenance are unknown; image-level development scores must not be presented as independent performance. This is not a physiological water-stress measurement or a drought/disease diagnosis.

## Start the whole-plant classifier

Use the tested **Python 3.13.0** environment. From this repository:

```sh
.venv-py313/bin/python run_wholeplant.py
```

Open **http://127.0.0.1:5001**. Upload a JPEG, PNG or WebP, choose **Analyze image**, inspect the mask and read any framing/background warnings. The launcher loads the versioned model named in `wholeplant_deployment_v1.json`. It does not train during requests. It uses port 5001 by default; use `PORT=5002` if that port is occupied.

Recreate the environment with CPython **3.13.0**, not merely any Python 3.13 patch:

```sh
python3.13 --version
python3.13 -m venv .venv-py313
.venv-py313/bin/python -m pip install -r requirements-py313.lock.txt
.venv-py313/bin/python -m pip check
.venv-py313/bin/python run_wholeplant.py
```

The strict artifact contract checks Python and library versions. Exact core versions: NumPy 2.4.6, OpenCV 5.0.0.93, scikit-learn 1.9.0, SciPy 1.18.0, pandas 3.0.6, joblib 1.5.3, Pillow 12.3.0, Flask 3.1.3; Matplotlib 3.11.2 is also pinned. Tests use pytest 9.1.1. Full dependency versions and historical Python 3.14.5 tests are in [environment.md](docs/environment.md). The supplied new model was trained under Python 3.13.0 and is intentionally refused under 3.14.5; retraining there creates a different compatible artifact.

Preview without loading a classifier:

```sh
LECOR_PREVIEW_ONLY=true .venv-py313/bin/python run_wholeplant.py
```

## What is implemented

1. UUID-based temporary uploads, format/decode checks, 10 MiB request and 16-million-pixel limits, automatic cleanup, JSON errors.
2. Aspect-preserving downsize to a maximum side of 800 pixels. HSV plus green-channel ratios select candidate foliage; opening and small-component removal retain multiple foliage regions and holes.
3. Reject unreadable, very small, uniform or insufficient-foliage images. Frame contact is a warning in **whole-plant mode**, not evidence that all leaves are visible. Multiple components are expected. The original single-leaf thresholds are unchanged.
4. One shared 24-feature extractor: HSV statistics, RGB means, GLI, normalized R/G chromaticity, LAB a*/b* mean/std, yellow/dark fractions, and upper/middle/lower GLI/yellow fractions. No individual-leaf contour, curl, turgor or leaf-count claim.
5. A regularized Random Forest was selected among LR, calibrated RBF SVM and RF using full-precision image-level development Macro F1. Preprocessing stays within CV pipelines. A provisional top-class probability below 0.60 returns **Uncertain / unsupported input** without a categorical result or indicator. High confidence does not establish input validity or scientific correctness.
6. Classified results show probabilities, other models' predictions, color maps, feature comparisons and cautious guidance. The LR coefficient explanation is available only when LR is the primary model; the deployed RF does not receive a fabricated coefficient explanation.
7. The optional indicator is an **arbitrary category display**: Healthy 10, Moderate 50, Severe 90. It conveys no information beyond the category and is not calibrated severity or confidence. Disease-like/confounded results have no indicator.

Original dataset/model/CSV/benchmark and backups remain unchanged. The old single-leaf path still exists via `app.py`; its incompatible original model is intentionally refused outside preview. Historical leaf-only instructions: [legacy_leaf_workflow.md](docs/legacy_leaf_workflow.md).

## New data and development results

New data: `dataset_wholeplant_v1_20261013/`, copied from the supplied Drive folder. There are 44 decodable byte-unique images and all pass the deliberately limited foliage gate. Passing does **not** verify correct labels, complete-plant framing or a clean mask. Photos mix isolated plants, foliage close-ups, gardens and different backgrounds. Automatic dHash screening found no pairs within distance 8, but this does not rule out related plants, sessions or near-duplicates.

The active artifact directory is recorded in `wholeplant_deployment_v1.json` and contains `manifest.csv`, `features.csv`, `model.joblib`, and readable/JSON development reports. Fill in real **plant_id, session_id, source, device, background_type, label_method and notes** in a copy of the manifest. `leaf_id` is optional for whole-plant photos. Do not invent IDs to make CV possible. Group-independent evaluation remains blocked until provenance and enough independent plants exist.

See [wholeplant_implementation.md](docs/wholeplant_implementation.md) for approach tradeoffs, results, scope changes and demo guidance. These are development diagnostics with model-selection bias, not final accuracy. The current data do not support dependable Moderate/Severe separation.

## Reproduce training and software checks

```sh
.venv-py313/bin/python train_wholeplant.py --development-only
.venv-py313/bin/python record_test_run.py
.venv-py313/bin/python verify_wholeplant.py
```

Training creates a new timestamped artifact directory and never silently replaces the deployment pointer. It checks cross-label exact-duplicate conflicts, deduplicates within a label, requires five accepted unique images per class, runs fixed 5-fold stratified image-level development CV, and then fits the three candidates on all accepted images. Scaling and SVM calibration occur inside each training fold. It does not use filled manifest identities for grouped validation: use a separately reviewed grouped protocol after provenance is supplied. Candidate parameters are fixed, and ties use balanced accuracy followed by LR/SVM/RF priority.

To deploy a later model, review its report and update the paths in `wholeplant_deployment_v1.json` to its artifact directory. Only load trusted local joblib files. Existing models enforce feature order, schema/extractor/preprocessing versions, class alignment, runtime versions and provenance metadata.

The test-client timing tool includes extraction and inference on training images; it is **software verification only**, not an accuracy test and not a browser/network benchmark. Camera capture, printing, keyboard usability and Chrome/Firefox/Safari/Edge still need [manual device verification](docs/manual_verification_checklist.md).
