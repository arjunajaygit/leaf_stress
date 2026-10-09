# Stage 5 — Features, provenance and explanations

Verified 2026-10-09. Production feature definitions/order, the original model, original CSV/benchmark and dataset are unchanged. No production model was selected or saved. No quality thresholds changed.

## Dependency environment

The test environment uses CPython 3.14 and a project-local `.venv-stage5` created with system-site-packages. Only NumPy was installed locally; the other tested packages were inherited from the existing installation. Global packages were not changed. `pip check` reports no broken requirements. This is not a claim of clean-install testing on every platform.

| Package | Pinned version | Reason |
|---|---|---|
| NumPy | 2.4.6 | Original loading under 2.5.1 emitted joblib array-shape deprecations; 2.4.6 loaded the unchanged artifact with zero warnings |
| joblib | 1.5.3 | Existing version, verified with NumPy 2.4.6 |
| scikit-learn | 1.9.0 | Retained installed estimator implementation to avoid an unverified serialization migration |
| OpenCV | 5.0.0.93 | Retained installed image-processing implementation |
| pandas | 3.0.6 | Retained installed tabular implementation |
| Flask | 3.1.3 | Retained installed web implementation |
| matplotlib | 3.11.2 | Retained installed visualization implementation |
| Pillow | 12.3.0 | Retained installed image-validation implementation |
| SciPy | 1.18.0 | Explicitly pinned the numerical dependency used by scikit-learn |
| pytest (development) | 9.1.1 | Retained test runner |

`requirements.txt` pins runtime packages; `requirements-dev.txt` pins pytest. The tests explicitly assert zero warnings while deserializing the original model under NumPy 2.4.6. Separate SVC `probability=True` training deprecations remain in historical training/reproduction tests; warnings are not hidden. Grouped SVM evaluation now leaves the deprecated probability argument unspecified, preserving its decision-only evaluation behavior.

## Model contract and training

`model_contract.py` defines the production feature schema, schema version, extractor version and validation. The unchanged 15 feature names remain the only production inputs. New CSV metadata carries extractor/schema/preprocessing versions. The training writer verifies those versions and records:

- Ordered feature schema and schema version.
- Extractor version, preprocessing version/settings and mask settings.
- SHA-256 of exact training feature-CSV bytes, with its definition; source manifest hash when available.
- Python, NumPy, SciPy, scikit-learn, joblib, OpenCV, pandas and Pillow versions.
- Timezone-aware training date and the actual CV protocol. The historical trainer explicitly records image-level development CV, not independent performance.

The application validates the contract **before activating any estimator**, including every comparison estimator's feature/class order. Missing versions, incompatible schemas or runtime libraries are refused. Exact runtime versions are intentionally conservative; use the recorded training environment when deploying. The dataset hash records provenance, not physiological validity. Only trusted pickle files may be read: these checks do not make pickle deserialization safe for untrusted input.

**Operational consequence:** the original artifact loads without serialization warnings but fails the new contract. Direct `python app.py` exits with a clear startup error. When served using Flask/WSGI, mask review remains available and passing images receive the existing “Model update required” screen with no prediction. The original artifact has not been retroactively annotated as compatible. There is no new valid production model to activate while data readiness remains blocked.

## Logistic Regression explanation

`explain_logistic()` uses the exact fitted StandardScaler and the predicted class's Logistic Regression coefficients. It reports each standardized value, its coefficient, their product, the intercept and reconstructed class decision score. Tests verify the sum against `decision_function` across predicted classes.

The UI keeps dataset-average comparisons separate. The new panel states **“Model explanation, not a biological cause.”** A class score is not a class probability; probabilities depend on all scores. Correlated features can redistribute coefficients. Other estimator types receive an unavailable explanation, rather than invented contributions. Old results are cleared with the other output panels.

## Ablations

`ablate_features.py` compares five separately defined feature sets using the Stage 4 nested grouped evaluator and identical partitions. All scaling and candidate selection occur inside inner folds. Experimental pixel/shape extraction lives in the shared feature module but is never called by production feature extraction.

- Existing 15 features.
- Existing features without the redundant ExG mean.
- Existing features plus mean pixelwise normalized r/g.
- Existing features plus floating CIELAB a*/b* mean/std.
- Oriented short/long minAreaRect ratio replacing axis-aligned aspect ratio.

The [real-data report](../reports/ablation_2026-10-09_102728_c377f2fd.md) stops with **insufficient classes/groups for a credible generalization estimate**. No real-data folds/models were fitted. Accepted images remain Healthy 4, Moderate 0, Severe 0, Disease 7, with no documented groups.

The [synthetic report](../reports/ablation_synthetic_2026-10-09_102723_26266451.md) exercises all variants, three candidate algorithm families, and nested folds using 24 invented images across six invented groups. Its trivially separable synthetic labels are deliberately software fixtures; scores are not project performance or grounds to change production features. Companion JSON files contain runtime versions, fold memberships and detailed metrics. Outer comparisons remain exploratory: choosing features using them would require a new untouched test set for an unbiased final estimate. No legacy-CSV ablation is offered; that CSV cannot supply the proposed pixel-level features.

## Actual tests

- `.venv-stage5/bin/python -m pytest -q -p no:cacheprovider`: **107 passed**. Eleven remaining warnings concern SVC probability training, not NumPy/joblib loading.
- `node tests/frontend_stage1.cjs`: **passed**, including coefficient rendering, unavailable fallback and prior stage regressions.
- `.venv-stage5/bin/python -m pip check`: **no broken requirements**.
- Tests cover version/schema/runtime rejection, estimator order, saved metadata round-trip, actual training writer in a temporary synthetic fixture, API explanation, coefficient arithmetic, chromaticity/LAB/oriented-ratio math, inner/outer group isolation and real-data refusal before fitting.
- Preserved-artifact tests verify original hashes. Synthetic training tests save only temporary models, never production artifacts.

## Still required

Documented plant/session identities, label provenance and acceptable independent images are still missing. A validated replacement model, real ablation evidence and physiological claims cannot be supplied without that work. Production feature-set changes still require explicit approval. Stage 6 has not started.
