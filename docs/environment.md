# Tested environment and recreation

**No validated classifier exists yet. Classification results cannot be presented as performance.**

Actual host: macOS 26.5.1, Apple Silicon arm64. Node v24.15.0 was used only for the frontend regression script. No Linux/Windows/browser compatibility is inferred.

| Environment | Construction | Actual result |
|---|---|---|
| CPython 3.14.5 | `.venv-stage5` with system-site-packages; project-local NumPy 2.4.6, other packages inherited | 134 pytest cases passed; frontend and pip check passed |
| CPython 3.13.0 | Fresh `.venv-py313` without system-site-packages; installed `requirements-dev.txt` | 134 pytest cases passed; frontend and pip check passed |

These exact combinations are known to work for the recorded software suite. Python 3.13 provides an independently installed second environment. Other patch/minor versions are not claimed tested. The strict saved-model contract includes exact Python/library versions; do not move a trained artifact between environments and assume compatibility.

## Recreate the clean Python 3.13 environment

From the repository root, with CPython 3.13.0 installed:

```sh
python3.13 --version
python3.13 -m venv .venv-py313
source .venv-py313/bin/activate
python -m pip install -r requirements-py313.lock.txt
python -m pip check
python record_test_run.py
LECOR_PREVIEW_ONLY=true python app.py
```

The original clean test used `requirements-dev.txt`; the lock file now records the exact resolved direct and transitive versions. Use a new empty virtual environment if recreating it. Global package changes are unnecessary.

For the tested Python 3.14.5 dependency combination:

```sh
python3.14 --version
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-py314.lock.txt
python -m pip check
python record_test_run.py
LECOR_PREVIEW_ONLY=true python app.py
```

The recorded 3.14 run used an environment with inherited packages; a clean 3.14 installation from this lock has not separately been tested. The 3.13 run is the clean-install evidence. Install the recorded Node version separately to reproduce the frontend test; Node is not required by Flask.

## Exact package versions

| Package | Python 3.13.0 | Python 3.14.5 |
|---|---|---|
| blinker | 1.9.0 | 1.9.0 |
| click | 8.5.0 | 8.4.2 |
| contourpy | 1.4.0 | 1.4.0 |
| cycler | 0.12.1 | 0.12.1 |
| flask | 3.1.3 | 3.1.3 |
| fonttools | 4.66.1 | 4.66.1 |
| iniconfig | 2.3.1 | 2.3.0 |
| itsdangerous | 2.2.0 | 2.2.0 |
| jinja2 | 3.1.6 | 3.1.6 |
| joblib | 1.5.3 | 1.5.3 |
| kiwisolver | 1.5.1 | 1.5.1 |
| markupsafe | 3.0.4 | 3.0.3 |
| matplotlib | 3.11.2 | 3.11.2 |
| narwhals | 2.26.0 | 2.24.0 |
| numpy | 2.4.6 | 2.4.6 |
| opencv-python | 5.0.0.93 | 5.0.0.93 |
| packaging | 26.3 | 26.2 |
| pandas | 3.0.6 | 3.0.6 |
| pillow | 12.3.0 | 12.3.0 |
| pluggy | 1.6.0 | 1.6.0 |
| pygments | 2.21.0 | 2.20.0 |
| pyparsing | 3.3.3 | 3.3.2 |
| pytest | 9.1.1 | 9.1.1 |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 |
| scikit-learn | 1.9.0 | 1.9.0 |
| scipy | 1.18.0 | 1.18.0 |
| six | 1.17.0 | 1.17.0 |
| threadpoolctl | 3.7.0 | 3.6.0 |
| werkzeug | 3.1.9 | 3.1.9 |

NumPy 2.4.6 + joblib 1.5.3 loads the original artifact with zero deserialization warnings in both tested environments. This does not activate the original model: its schema/preprocessing metadata is incompatible. Existing SVC probability-training warnings remain (11 per suite). No warnings are suppressed by the application.

## Current runtime modes

`LECOR_PREVIEW_ONLY=true` skips model loading and disables every classification, probability and indicator path. It starts even with missing/corrupt model files. The banner, masks, provisional reasons and passing-image color/GLI previews are available. `/api/benchmark` withholds historical numbers in this mode.

Normal mode (`LECOR_PREVIEW_ONLY=false`, default) requires a compatible versioned model for direct startup. It currently exits clearly for the original artifact. A future compatible model must be explicitly selected with `LECOR_MODEL_PATH` and matching benchmark configuration; metadata compatibility alone does not establish validated performance.

Default URL: `http://127.0.0.1:5001`. Debug is off. Set `PORT` for another local port. Intentional LAN binding uses `LECOR_HOST`; secure hosting and phone/device acceptance are separate work.
