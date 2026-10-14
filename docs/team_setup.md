# Clone the advaith branch with the same classifier and SAM

## Windows x64 quick start

Install Git and **CPython 3.13.15 (64-bit)**. This exact Python patch is required
by the currently selected classifier, not just any Python 3.13 installation.
Node is optional and is only needed for frontend tests.

```powershell
git clone --branch advaith --single-branch https://github.com/arjunajaygit/leaf_stress.git
cd leaf_stress
py -3.13 --version
```

Confirm the printed version is **3.13.15**, then:

```powershell
py -3.13 -m venv .venv-py313
.\.venv-py313\Scripts\python.exe setup_project.py
.\.venv-py313\Scripts\python.exe run_wholeplant.py
```

`setup_project.py` checks the selected classifier and Python version, installs the
full pinned Windows CPU dependency lock, downloads and verifies the official SAM
ViT-B checkpoint, checks package consistency, and validates classifier loading.
It does not retrain or replace the classifier. Internet is needed for setup; the
checkpoint is about 375 MB and PyTorch's wheel about 216 MB, plus dependencies.
The model weights and virtual environment are intentionally excluded from Git.

Open http://127.0.0.1:5001. For an existing checkout, stop the old server and refresh
the browser with Ctrl+F5 after pulling. See [SAM marking instructions](sam_setup.md).

Check an existing installation without downloading or installing anything:

```powershell
.\.venv-py313\Scripts\python.exe setup_project.py --check-only
```

For an existing checkout, stop the server and run:

```powershell
git switch advaith
git pull --ff-only origin advaith
.\.venv-py313\Scripts\python.exe setup_project.py
.\.venv-py313\Scripts\python.exe run_wholeplant.py
```

## Camera and result display

Choose **Take a plant photo**, allow browser camera access, capture, review, then
choose **Use photo**. The captured JPEG follows the same crop, SAM and analysis
workflow as an uploaded image. No microphone is requested, and capture alone does
not send the photo for analysis. Camera capture adds no Python dependencies.

Live browser camera access requires localhost or HTTPS. Opening the app on a phone
using a plain HTTP LAN address may prevent live capture; use **Device camera /
photo picker**, or serve the app through HTTPS. Real phone camera behaviour still
needs testing on the target device; automated camera checks use simulated video.

The whole-plant display threshold defaults to 0.50. Estimates below 0.60 are labelled
tentative; below 0.50 the app shows no clear result. An existing
`LECOR_MIN_CLASS_PROBABILITY` environment override changes this behaviour. These
are experimental visual estimates, not measurements of plant water status.

The deployment pointer still selects `wholeplant_v1_20261013T144610Z_b344aa77`.
The newer `wholeplant_v1_20261014T071137Z_0f1f368f` training artifact is included
for reference but is not automatically selected. Updating dataset images does not
change the running classifier; retraining and an explicit deployment update are
needed to change the model.

## What must be in the commit

- Application changes, `result_presentation.py`, `static/plant_camera.js`,
  `static/plant_crop.js`, `static/plant_sam.js`, tests and docs.
- `sam_segmentation.py`, `setup_sam.py`, `setup_project.py`, `verify_sam.py` and both
  SAM requirements files, including `requirements-windows-sam.lock.txt`.
- `wholeplant_deployment_v1.json` **and the entire artifact folder it names**:
  `artifacts/wholeplant_v1_20261013T144610Z_b344aa77/`. Without the new model and
  report, cloning the changed deployment pointer cannot launch the classifier.
- Added dataset images if teammates should reproduce training. They are not
  required for inference, but the real SAM smoke check uses `Healthy/healthy_001.JPG`.

Do not commit `.venv-py313`, credentials, user uploads, `models/*.pth` or partial
downloads. The Git ignore rules exclude these runtime files.

## Verify the feature

```powershell
.\.venv-py313\Scripts\python.exe -m pytest tests/test_stage3.py tests/test_presentation.py tests/test_sam.py tests/test_wholeplant.py -q
node tests/frontend_stage1.cjs
node tests/frontend_sam.cjs
.\.venv-py313\Scripts\python.exe verify_sam.py
```

The last command uses the actual SAM checkpoint through Flask and records outputs
under `reports/sam_smoke_*`. First CPU use may take tens of seconds. Browser/device
behaviour still needs manual verification.

## What “same result” means

Use the same branch commit, selected classifier, Python/library versions, image,
applied crop, and SAM keep/exclude points. Different clicks select different pixels
and can change the classification; minor numerical variation across hardware is
also possible. SAM exports include a selection record so image preparation can be
documented. An uncertain result below the provisional whole-plant 0.50 threshold is expected
behaviour, not a setup failure. This classifier was trained on 54 accepted original
photos, not a consistently SAM-prepared dataset; SAM does not automatically improve
its diagnostic quality.

## Other platforms or Python versions

The combined lock is the tested Windows x64 CPU environment. macOS/ARM and other
Python versions have not been checked for this deployment. Do not install the
Windows CPU lock there and assume equivalence. Use compatible platform-specific
PyTorch packages, reproduce the classifier's exact core runtime where available,
or retrain a separate compatible artifact and explicitly update its deployment
pointer. The Python 3.13.0/macOS instructions in older reports refer to the earlier
44-image experiment, not the current deployment.
