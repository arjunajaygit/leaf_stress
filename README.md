# Lecor — Camera-Based Crop Leaf Water-Stress Detection

Lecor is an academic prototype for **visual leaf classification** using OpenCV, scikit-learn and Flask. Its goal is to investigate whether RGB color and projected shape features can help flag possible water-stress symptoms. It does not establish the cause of symptoms, physiological water status or validated early-warning performance.

## Current scope and limitations

The supplied dataset contains 56 image files and 40 byte-unique images: Healthy 8, Moderate Water Stress 9, Severe Water Stress 11 and Disease Confounded 12. Plant/session identities, label provenance and independent physiological measurements are missing. Background and framing vary with class, so the model may learn acquisition conditions instead of relevant symptoms.

With a compatible trained model, the app predicts one of the four existing dataset labels. `Disease_Confounded` is displayed as **“Disease-like or other confounding symptoms suspected”**, not a confirmed disease diagnosis. Other class names are dataset categories, not verified physiological assessments. Drought and disease can coexist.

Use one fully visible leaf on a plain contrasting background with even lighting. The current fixed-threshold segmentation can merge green backgrounds and multiple leaves or accept non-leaf objects. Provisional mask-quality checks now reject border contact, excessive/insufficient foreground, multiple large components and tiny outlines. A provisional top-class-probability threshold (default 0.60) produces “Uncertain / unsupported input” without a category or indicator. These checks do not establish leaf identity; a centered green non-leaf object can still pass.

This is a small-dataset, visual-only prototype, not a replacement for direct soil assessment, laboratory testing or expert advice.

## What the code does

1. Validates an uploaded JPEG, PNG or WebP, request size and image dimensions.
2. Resizes with preserved aspect ratio to longest side 800 px (area interpolation for reduction, linear for enlargement), then segments with a fixed HSV range and resolution-scaled opening/closing (5×5 at 800 px). It selects/fills the largest external contour and checks all candidate components before allowing feature extraction. Upsampling smaller photos does not add detail.
3. Uses one shared implementation to extract 15 numerical features for training and inference.
4. Requires a model with matching preprocessing metadata, then runs the selected classifier and applies provisional confidence rejection. Predictions from other fitted models are shown separately only for classified results.
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

**Feature measurements vs. dataset averages** shows all available feature values, pooled all-class means and standardized deviations `(value − mean) / std`. These are descriptive comparisons, not contributions to the prediction or biological causes. No Random Forest importance weighting is applied. The stored averages are not a healthy reference population. A separate Logistic Regression model-explanation panel uses the fitted StandardScaler values and the predicted class coefficients. Each term is coefficient × standardized value; all terms plus the intercept reconstruct that class decision score. This is a model explanation, not a biological cause or a decomposition of probability. It is unavailable for other estimator types.

The color map divides masked pixels using fixed HSV rules into yellow-toned, dark/brown-toned and remaining pixels. The fractions do not diagnose chlorosis, necrosis or healthy tissue; illumination, shadows and incorrect masks can change them. Legacy internal keys such as `healthy_pct` and `pathology` are retained for compatibility with the processing code, with corrected visible labels. Display images are resized to a maximum side of 640 pixels. The GLI heatmap is an RGB visualization.

## Training and current evidence

`build_dataset.py` is import-safe, validates the manifest against image hashes, reports cross-label duplicate conflicts, removes byte-identical duplicate rows, recomputes quality checks and excludes rejected images. It reads documented plant/session groups and writes a versioned feature CSV plus a content-hashed preprocessing metadata sidecar. It stops before writing if any class has fewer than two accepted samples. Original images are never removed.

`train_model.py` compares Random Forest, scaled RBF SVM, Gradient Boosting, scaled Logistic Regression and scaled KNN using stratified image-level CV. It computes metrics from pooled out-of-fold predictions, selects the highest Macro F1, then fits all candidates on all development rows. Training does not occur during web requests. The trainer refuses CSVs with missing/mismatched preprocessing metadata, changed CSV content, missing classes or fewer than two accepted samples per class. This minimum only prevents invalid CV setup; it does not establish adequate sample size.

The original saved winner (now incompatible with resized preprocessing) is Logistic Regression: accuracy **0.6250**, Macro F1 **0.6019**, on image-level development CV. These are **not independent final test estimates**: model selection used those same results, and plant/session independence is unknown. No grouped evaluation or new scientific performance result has been produced by the software fixes. Stages 3–4 have not retrained or activated a new production model. The optional legacy reproduction is not gate-checked and must not be reported as performance. The saved benchmark is accessible at `GET /api/benchmark`; there is no benchmark dashboard in the current UI.


## Dataset manifest and grouped evaluation

`dataset_manifest.csv` contains all 56 files, with 40 distinct hashes and 16 duplicate copies. It includes dimensions and Stage 3 gate status/foreground fraction. Border-contact fraction is an additional perimeter measurement from the same masks; it does not change the gate. See [the manifest guide](docs/dataset_manifest_guide.md) for its definition and safe regeneration.

**Fill in plant/leaf/session IDs, source, device, background type and label method from actual records. These fields are currently blank. Do not infer identities or labels from filenames.** The [collection protocol](docs/data_collection_protocol.md) provides suggested per-class targets and a collector table.

```sh
python3 evaluate_grouped.py
# Optional historical diagnostics only — NEVER report as performance:
python3 evaluate_grouped.py --legacy-development-only --include-baselines
```

The default evaluation uses only gate-passing, byte-unique images. It currently exits with status 2 and **“insufficient classes/groups for a credible generalization estimate”**, records per-class deficits, and fits no models: Healthy 4, Moderate 0, Severe 0, Disease Confounded 7, with no documented groups. JSON and Markdown reports use unique `grouped_eval_<date>_...` filenames. Legacy diagnostics use separate `legacy_development_only_<date>_...` filenames and the unchanged original 40-row CSV; they are image-level, not group-independent and not gate-checked.

When sufficient documented data exists, the evaluator preflights all partitions of 3-fold outer / 3-fold inner StratifiedGroupKFold. It requires at least five documented groups per class as a feasibility guard, not a statistically derived sample-size recommendation. Grouping prefers plant IDs with conservative session fallback; missing IDs block evaluation. Session fallback cannot establish independence of unknown plants across sessions.

Inside each outer training partition it compares scaled Logistic Regression, scaled RBF SVM and a regularized Random Forest. KNN and Gradient Boosting are optional baselines. Scalers are fitted inside pipelines. Inner selection uses full-precision mean Macro F1, then balanced accuracy, fixed model priority and candidate order. Outer predictions estimate the entire selection procedure; they must not be reused to select a production winner and claim an unbiased final result. No production model is saved. Reports include class metrics, confusion matrices and specified error counts when evaluation is feasible. Training/evaluation never runs during requests.

## Stage 3 quality gate and model compatibility

The original model was trained on original-resolution features. It is preserved but **cannot be used for classification with the new resized extractor**. The original artifact is now refused at load time because it lacks the versioned schema/provenance contract. Direct `python3 app.py` startup exits with a clear error until a compatible model is explicitly configured. When served through a WSGI/Flask runner, the UI remains available for mask review: passing images receive “Model update required” (HTTP 503), and rejected images receive a mask preview, all reason codes and a retake message (HTTP 422). No incompatible estimator is activated. No class/indicator is returned for rejected, uncertain or model-incompatible inputs.

`reports/mask_quality_report.csv` contains one row per byte-unique image, duplicate aliases, both selected/candidate foreground fractions, border contact, component counts, contour area, working dimensions, pass/fail and all reason codes. `reports/mask_quality_summary.md` includes overlapping and primary failure counts, a threshold snapshot and the rejected-image list.

The current report contains **11 passes and 29 failures**. All 29 failures touch the frame; 9 also exceed 95% foreground. Accepted classes are Healthy 4 and Disease Confounded 7; **no Moderate or Severe images pass**. A new valid four-class model cannot be trained from this accepted subset. Collect appropriate examples, or review masks and acquisition assumptions before approving any threshold change. Do not relax thresholds solely to recover class counts.

| Setting in `config.py` | Provisional default |
|---|---|
| `WORKING_LONGEST_SIDE` | 800 px |
| `MORPH_KERNEL_AT_REFERENCE` / `MORPH_REFERENCE_SIDE` | 5 / 800; scaled to an odd kernel |
| `MASK_REJECT_BORDER_CONTACT` | True (selected mask) |
| `MASK_MIN_FOREGROUND_FRACTION` / `MASK_MAX_FOREGROUND_FRACTION` | 0.02 / 0.95 |
| `MASK_MIN_CONTOUR_AREA_FRACTION` / `MASK_MIN_CONTOUR_AREA_PIXELS` | 0.005 / 128; require the larger minimum |
| `MASK_LARGE_COMPONENT_MIN_FRACTION` | 0.01 of image pixels |
| `MASK_LARGE_COMPONENT_RELATIVE_FRACTION` | 0.20 of the largest component; both component cutoffs must be met |
| `MASK_MAX_LARGE_COMPONENTS` | 1 |
| `MIN_CLASS_PROBABILITY` | 0.60; configurable using `LECOR_MIN_CLASS_PROBABILITY` |

The confidence check uses probabilities before rounding. It is uncalibrated and provisional, not evidence of reliability. Missing probability support is a model error. Both the original model and its benchmark are retained for historical reference; `/api/benchmark` still serves that original development benchmark by default.

Reproduce the mask audit without overwriting earlier reports by choosing new filenames:

```sh
python3 report_mask_quality.py --csv reports/mask_quality_report_review2.csv --summary reports/mask_quality_summary_review2.md
```

The report script refuses existing output paths. New model bundles include feature schema and schema version, extractor and preprocessing versions/settings, mask settings, exact feature-CSV SHA-256, library/Python versions, timezone-aware training date and CV protocol. Loading rejects missing/mismatched schemas, versions, estimator feature/class order or runtime library versions. The CSV hash records training provenance; deployment does not require the original training CSV. Only trusted local pickle files may be loaded; metadata validation is not a pickle security sandbox.

## Run locally

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Requires an explicitly configured compatible model; original model is refused:
python app.py
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

The legacy `train_model.py` command uses image-level CV even when group columns exist; use `evaluate_grouped.py` for the grouped methodology. Training is currently blocked by missing accepted Moderate/Severe examples; the commands above describe the workflow once suitable data is available. Generated filenames contain timestamps and UUIDs; creation is exclusive. Relative configured paths resolve against the project directory, not the shell working directory. Defaults still read the original artifacts. `backup_original/` contains copies and a SHA-256 manifest; do not edit these or the original dataset/artifacts.

## Files and tests

| File | Responsibility |
|---|---|
| `app.py` | Routes, model loading, predictions and dataset comparisons |
| `feature_extraction.py` | Shared features, mask, color visualizations, heuristic and guidance |
| `config.py` | Paths, upload limits, preprocessing, quality thresholds and server settings |
| `report_mask_quality.py` | Read-only unique-image mask report and failure counts |
| `dataset_manifest.py` | Provenance template, hash validation, duplicate and group resolution |
| `build_dataset.py` | Import-safe, gate-passing versioned feature CSV and group metadata |
| `evaluate_grouped.py` | Guarded nested grouped evaluation and separate legacy diagnostics |
| `train_model.py` | Image-level development benchmark and fitted bundles |
| `templates/index.html` | Scan, pipeline, measurements and browser printing |
| `tests/` | Automated regression checks; not device acceptance evidence |
| `docs/` | Stage reports and proposed SRS wording |

```sh
python3 -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node tests/frontend_stage1.cjs
```

The JavaScript check uses a minimal DOM and the actual inline script. Node is needed only for this check, not to run the Flask app. Dependency pins and ablation tooling are implemented. Fuller academic test documents remain Stage 6 work. See `docs/stage5_report.md` for the tested dependency environment and actual validation results.

## Experimental feature ablations (Stage 5)

```sh
python ablate_features.py                 # real data: blocked, exit 2
python ablate_features.py --synthetic     # software validation only
```

The script compares the unchanged 15-feature baseline, baseline without ExG, baseline plus pixelwise normalized r/g means, baseline plus CIELAB a*/b* mean/std, and an oriented short/long axis ratio replacing axis-aligned aspect ratio. Additions are separate experiments, not cumulative. All comparisons reuse identical grouped outer folds; scaling and algorithm tuning remain inside inner folds. No feature variant is selected or saved for production. Comparing outer results to choose features would need a new untouched final test for an unbiased estimate.

The real dataset still lacks accepted Moderate/Severe images and documented groups. The synthetic report uses invented pixels, labels and plant IDs and is not evidence of project accuracy. No legacy-CSV ablation is offered because that CSV cannot recover the proposed pixel-level features.

NumPy is pinned to 2.4.6 with joblib 1.5.3: the original artifact deserialized with zero warnings in that tested combination. Other direct dependencies retain installed versions (including scikit-learn 1.9.0). Deserialization compatibility does not make the original artifact compatible with the current extractor. Existing legacy SVM probability training produces a separate scikit-learn deprecation warning; warnings are not suppressed.
