# Lecor — Camera-Based Crop Leaf Water-Stress Detection

Lecor is an academic prototype for **visual leaf classification** using OpenCV, scikit-learn and Flask. Its goal is to investigate whether RGB color and projected shape features can help flag possible water-stress symptoms. It does not establish the cause of symptoms, physiological water status or validated early-warning performance.

## Current scope and limitations

The supplied dataset contains 56 image files and 40 byte-unique images: Healthy 8, Moderate Water Stress 9, Severe Water Stress 11 and Disease Confounded 12. Plant/session identities, label provenance and independent physiological measurements are missing. Background and framing vary with class, so the model may learn acquisition conditions instead of relevant symptoms.

The app predicts one of the four existing dataset labels. `Disease_Confounded` is displayed as **“Disease-like or other confounding symptoms suspected”**, not a confirmed disease diagnosis. Other class names are dataset categories, not verified physiological assessments. Drought and disease can coexist.

Use one fully visible leaf on a plain contrasting background with even lighting. The current fixed-threshold segmentation can merge green backgrounds and multiple leaves or accept non-leaf objects. Automatic mask quality rejection and low-confidence abstention are not implemented yet. These remain scheduled for Stage 3.

This is a small-dataset, visual-only prototype, not a replacement for direct soil assessment, laboratory testing or expert advice.

## What the code does

1. Validates an uploaded JPEG, PNG or WebP, request size and image dimensions.
2. Segments at original resolution with a fixed HSV range, 5×5 opening/closing and the largest external contour, filling its interior.
3. Uses one shared implementation to extract 15 numerical features for training and inference.
4. Runs the selected saved classifier. Predictions from other fitted models are shown separately on the same image.
5. Produces an optional experimental heuristic indicator, color-threshold visualizations, descriptive feature comparisons and cautious inspection guidance.
6. Displays results in the browser; the print button opens the browser print dialog.

### Features

- HSV means and standard deviations: `h_mean`, `h_std`, `s_mean`, `s_std`, `v_mean`, `v_std`.
- RGB channel means: `r_mean`, `g_mean`, `b_mean` (OpenCV's BGR input is unpacked correctly).
- Mean pixelwise ExG: `2G − R − B`; exactly redundant with the three RGB means.
- Mean pixelwise GLI: `(2G − R − B) / (2G + R + B + 1e-6)`.
- **2D projected shape descriptors:** solidity `A/hull_area`, circularity `4πA/P²`, extent `A/(w*h)`, and axis-aligned aspect ratio `w/h`.

These shape descriptors do not directly measure curl, turgor or water content. Perspective, natural leaf shape, clipping and segmentation affect them. RGB values and indices are camera-dependent color descriptors, not calibrated reflectance, chlorophyll or hydration measurements. Area and perimeter are intermediate calculations, not independent model inputs.

### Experimental relative visual-stress indicator

For non-confounded predictions, the app reports a rounded integer on 0–100. It is a **heuristic, not a physiological measurement or a class probability**. It has no established agronomic interpretation or validated severity cutoffs.

The existing formula is retained for continuity:

```text
base = 10*pHealthy + 55*pModerate + 92*pSevere + 45*pDisease
shape_adjustment = max(0, (0.95-solidity)*30) + max(0, (0.50-compactness)*20)
color_adjustment = max(0, (0.20-GLI)*50)
indicator = round(clip(base + 0.25*shape_adjustment + 0.25*color_adjustment, 0, 100))
```

The weights and color thresholds are arbitrary prototype choices, not fitted physiological relationships. Disease probability still contributes an arbitrary term to non-disease results; this remains a limitation. Shape and greenness may duplicate evidence used by the classifier. The API returns `null`, and the UI hides the indicator, for a `Disease_Confounded` prediction. Its old name has been retired because the formula is not a validated crop water stress index.

### Measurements and visualizations

**Feature measurements vs. dataset averages** shows all available feature values, pooled all-class means and standardized deviations `(value − mean) / std`. These are descriptive comparisons, not contributions to the prediction or biological causes. No Random Forest importance weighting is applied. The stored averages are not a healthy reference population. Model-specific explanations are planned for Stage 5.

The color map divides masked pixels using fixed HSV rules into yellow-toned, dark/brown-toned and remaining pixels. The fractions do not diagnose chlorosis, necrosis or healthy tissue; illumination, shadows and incorrect masks can change them. Legacy internal keys such as `healthy_pct` and `pathology` are retained for compatibility with the processing code, with corrected visible labels. Display images are resized to a maximum side of 640 pixels. The GLI heatmap is an RGB visualization.

## Training and current evidence

`build_dataset.py` removes byte-identical duplicates and writes a versioned CSV. It currently executes when imported; import safety and label-conflict handling are planned for Stage 4. Do not import it as a library yet.

`train_model.py` compares Random Forest, scaled RBF SVM, Gradient Boosting, scaled Logistic Regression and scaled KNN using stratified image-level CV. It computes metrics from pooled out-of-fold predictions, selects the highest Macro F1, then fits all candidates on all development rows. Training does not occur during web requests.

The original saved winner is Logistic Regression: accuracy **0.6250**, Macro F1 **0.6019**, on image-level development CV. These are **not independent final test estimates**: model selection used those same results, and plant/session independence is unknown. No grouped evaluation or new scientific performance result has been produced by the software fixes. The saved benchmark is accessible at `GET /api/benchmark`; there is no benchmark dashboard in the current UI.

## Run locally

```sh
python3 -m pip install -r requirements.txt
python3 app.py
```

Open [localhost:5001](http://localhost:5001). Default binding is `127.0.0.1`; debug is off. For intentional LAN access, set `LECOR_HOST=0.0.0.0`. Camera availability through the file picker depends on the device/browser; there is no explicit webcam capture implementation. Camera capture, cross-browser operation and printed report completeness still need manual verification. Inactive tabs are not automatically included in printing.

Configuration is in `config.py`. Upload defaults are 10 MiB per request and 16 million decoded pixels, configurable through `LECOR_MAX_UPLOAD_BYTES` and `LECOR_MAX_IMAGE_PIXELS`. Uploads receive UUID names in temporary directories and are removed after processing. Model failures produce startup logs and prediction-service errors. Direct CLI startup stops if the model cannot load. Only load trusted joblib artifacts.

### Generate new artifacts without replacing originals

```sh
python3 build_dataset.py
# Use the actual versioned filename printed by the builder:
LECOR_CSV_PATH=leaf_features_v1_REPLACE_WITH_GENERATED_ID.csv python3 train_model.py
# Opt into matching new artifacts explicitly:
LECOR_MODEL_PATH=leaf_stress_model_v1_REPLACE_WITH_RUN_ID.pkl \
LECOR_BENCHMARK_PATH=model_benchmark_v1_REPLACE_WITH_RUN_ID.json python3 app.py
```

Generated filenames contain timestamps and UUIDs; creation is exclusive. Relative configured paths resolve against the project directory, not the shell working directory. Defaults still read the original artifacts. `backup_original/` contains copies and a SHA-256 manifest; do not edit these or the original dataset/artifacts.

## Files and tests

| File | Responsibility |
|---|---|
| `app.py` | Routes, model loading, predictions and dataset comparisons |
| `feature_extraction.py` | Shared features, mask, color visualizations, heuristic and guidance |
| `config.py` | Paths, upload limits and server settings |
| `build_dataset.py` | Deduplication and versioned feature CSV |
| `train_model.py` | Image-level development benchmark and fitted bundles |
| `templates/index.html` | Scan, pipeline, measurements and browser printing |
| `tests/` | Automated regression checks; not device acceptance evidence |
| `docs/` | Stage reports and proposed SRS wording |

```sh
python3 -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node tests/frontend_stage1.cjs
```

The JavaScript check uses a minimal DOM and the actual inline script. Node is needed only for this check, not to run the Flask app. Dependency pinning, mask gates, grouped evaluation, ablations and fuller academic test documents remain future approved stages.
