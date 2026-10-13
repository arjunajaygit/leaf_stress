# Lecor — experimental whole-plant visual assessment

A working **OpenCV + scikit-learn + Flask** demonstration now accepts a whole tomato-plant photo, extracts candidate foliage across the image, and returns a dataset-category prediction or an uncertainty outcome. It does not identify every leaf or assess hidden leaves/roots.

**A classifier is trained and runnable, but no scientifically validated classifier exists yet.** The current classifier uses 54 accepted images from 55 supplied files. Plant/session identities and label provenance are unknown; image-level development scores must not be presented as independent performance. This is not a physiological water-stress measurement or a drought/disease diagnosis.

## Start the whole-plant classifier

Follow [the complete teammate setup](docs/team_setup.md) for a fresh clone.
The active model requires **CPython 3.13.15**, with the current CPU SAM environment
checked on Windows x64. In PowerShell:

```powershell
py -3.13 --version
py -3.13 -m venv .venv-py313
.\.venv-py313\Scripts\python.exe setup_project.py
.\.venv-py313\Scripts\python.exe run_wholeplant.py
```

Open **http://127.0.0.1:5001**. Upload a JPEG, PNG or WebP, choose **Analyze image**, inspect the mask and read any framing/background warnings. The launcher loads the versioned model named in `wholeplant_deployment_v1.json`. It does not train during requests. It uses port 5001 by default; use `PORT=5002` if that port is occupied.

Confirm the Python version is **3.13.15** before creating the environment.
`setup_project.py` installs `requirements-windows-sam.lock.txt`, downloads and
verifies SAM weights (excluded from Git), and checks the selected classifier.
Use `setup_project.py --check-only` to validate without installing or downloading.

### Historical macOS setup (earlier 44-image artifact only)

The following Python 3.13.0 instructions refer to the earlier experiment, not the
current classifier. Use the teammate setup above for the current deployment.

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

### Select a target plant in a crowded photo

For trees directly behind the plant, use the optional **SAM** keep/exclude-point
selection after cropping. Review the overlay, then apply the selected region
before classification. See [SAM setup and training preparation](docs/sam_setup.md).

In whole-plant mode, drag a rectangle on the photo (touch also works), or enter
left/top/right/bottom percentages using the keyboard. Click **Apply crop**, then
**Preview foliage mask**. Adjust the rectangle if neighbouring plants dominate
the mask, and click **Analyze image** when satisfied. **Use full photo** restores
the original bounds. Only the applied crop is uploaded for analysis. Crop selection
does not separate overlapping plants inside the rectangle. Edge contact is a warning,
not proof that the target plant is clipped.

Use **Download selected image** to save the exact applied crop as a PNG for training.
Create a separate `dataset_wholeplant_cropped` folder with the same four class
subfolders, and save consistently selected crops there under unique filenames.
Keep original images separately; do not include both an original and its crop as
independent training examples. Record the original filename and real plant/session
identity in your provenance records. Train with:

```powershell
.\.venv-py313\Scripts\python.exe train_wholeplant.py --dataset dataset_wholeplant_cropped --development-only
```

Review the new manifest and report before updating `wholeplant_deployment_v1.json`.
The current model remains usable for experimentation, but it was not necessarily
trained with this crop protocol. Evaluate on crops from independent held-out plants.

1. UUID-based temporary uploads, format/decode checks, 10 MiB request and 16-million-pixel limits, automatic cleanup, JSON errors.
2. Aspect-preserving downsize to a maximum side of 800 pixels. HSV plus green-channel ratios select candidate foliage; opening and small-component removal retain multiple foliage regions and holes.
3. Reject unreadable, very small, uniform or insufficient-foliage images. Frame contact is a warning in **whole-plant mode**, not evidence that all leaves are visible. Multiple components are expected. The original single-leaf thresholds are unchanged.
4. One shared 24-feature extractor: HSV statistics, RGB means, GLI, normalized R/G chromaticity, LAB a*/b* mean/std, yellow/dark fractions, and upper/middle/lower GLI/yellow fractions. No individual-leaf contour, curl, turgor or leaf-count claim.
5. Logistic Regression was selected among LR, calibrated RBF SVM and RF using full-precision image-level development Macro F1 on 54 accepted images. Preprocessing stays within CV pipelines. A provisional top-class probability below 0.60 returns **Model ran; no confident classification**, with diagnostic model scores but no categorical result or indicator. High confidence does not establish input validity or scientific correctness.
6. Classified results show probabilities, other models' predictions, color maps, feature comparisons and cautious guidance. The LR coefficient explanation is available only when LR is the primary model.
7. The optional indicator is an **arbitrary category display**: Healthy 10, Moderate 50, Severe 90. It conveys no information beyond the category and is not calibrated severity or confidence. Disease-like/confounded results have no indicator.

Original dataset/model/CSV/benchmark and backups remain unchanged. The old single-leaf path still exists via `app.py`; its incompatible original model is intentionally refused outside preview. Historical leaf-only instructions: [legacy_leaf_workflow.md](docs/legacy_leaf_workflow.md).

## New data and development results

Current deployment: `artifacts/wholeplant_v1_20261013T144610Z_b344aa77/`, trained
under CPython 3.13.15. Logistic Regression; development Macro F1 0.4289, accuracy
0.4259. The current report records 55 files and 54 accepted unique images:
Healthy 14, Moderate 13, Severe 14, Disease Confounded 13. The added AVIF was not
accepted by this training run. The classifier has not been retrained on
consistently SAM-prepared images. These are development diagnostics, not
independent performance. The paragraphs below describe the original dataset.

New data: `dataset_wholeplant_v1_20261013/`, copied from the supplied Drive folder. There are 44 decodable byte-unique images and all pass the deliberately limited foliage gate. Passing does **not** verify correct labels, complete-plant framing or a clean mask. Photos mix isolated plants, foliage close-ups, gardens and different backgrounds. Automatic dHash screening found no pairs within distance 8, but this does not rule out related plants, sessions or near-duplicates.

The active artifact directory is recorded in `wholeplant_deployment_v1.json` and contains `manifest.csv`, `features.csv`, `model.joblib`, and readable/JSON development reports. Fill in real **plant_id, session_id, source, device, background_type, label_method and notes** in a copy of the manifest. `leaf_id` is optional for whole-plant photos. Do not invent IDs to make CV possible. Group-independent evaluation remains blocked until provenance and enough independent plants exist.

See [wholeplant_implementation.md](docs/wholeplant_implementation.md) for approach tradeoffs, results, scope changes and demo guidance. These are development diagnostics with model-selection bias, not final accuracy. The current data do not support dependable Moderate/Severe separation.

## Reproduce training and software checks

```powershell
.\.venv-py313\Scripts\python.exe train_wholeplant.py --development-only
.\.venv-py313\Scripts\python.exe -m pytest tests/test_stage3.py tests/test_sam.py tests/test_wholeplant.py -q
node tests/frontend_stage1.cjs
node tests/frontend_sam.cjs
.\.venv-py313\Scripts\python.exe verify_sam.py
```

Training creates a new timestamped artifact directory and never silently replaces the deployment pointer. It checks cross-label exact-duplicate conflicts, deduplicates within a label, requires five accepted unique images per class, runs fixed 5-fold stratified image-level development CV, and then fits the three candidates on all accepted images. Scaling and SVM calibration occur inside each training fold. It does not use filled manifest identities for grouped validation: use a separately reviewed grouped protocol after provenance is supplied. Candidate parameters are fixed, and ties use balanced accuracy followed by LR/SVM/RF priority.

To deploy a later model, review its report and update the paths in `wholeplant_deployment_v1.json` to its artifact directory. Only load trusted local joblib files. Existing models enforce feature order, schema/extractor/preprocessing versions, class alignment, runtime versions and provenance metadata.

The test-client timing tool includes extraction and inference on training images; it is **software verification only**, not an accuracy test and not a browser/network benchmark. Camera capture, printing, keyboard usability and Chrome/Firefox/Safari/Edge still need [manual device verification](docs/manual_verification_checklist.md).
