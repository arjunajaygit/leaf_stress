# Stage 4 — Dataset manifest and grouped evaluation

Verified 2026-10-09. This stage preserves the original dataset, feature CSV, model and benchmark, and does not alter any quality-gate thresholds or production feature definitions.

## Delivered

- A 56-row manifest covering 40 byte-unique images and 16 duplicate copies, with SHA-256, dimensions, duplicate links, original class folders and gate diagnostics. Gate status and foreground fraction match the Stage 3 snapshot. The new border-perimeter fraction is measured from the same selected masks; it is diagnostic only.
- An import-safe builder that validates hashes, rejects cross-label duplicate conflicts, recomputes gates, reads documented groups and fails before output when classes are too small. New artifacts use versioned filenames.
- A nested grouped evaluator with scaling inside pipelines, full-precision Macro F1 selection and deterministic tie-breaks, conservative plant/session grouping, and class/group/hash checks on every inner and outer partition before fitting. Optional baselines and separate legacy diagnostics are available.
- Practical collection targets and a collector table. Targets are suggested, not statistically derived.
- The model-update feedback button now reads “Choose another photo”; quality rejection still says “Retake photo”.

## Actual dataset outcome

The committed grouped readiness report is [here](../reports/grouped_eval_2026-10-08_191647_0697e39b.md), with full details in its companion JSON. It was generated on October 8; this final verification occurred October 9.

| Class | Accepted unique images | Documented groups |
|---|---:|---:|
| Healthy | 4 | 0 |
| Moderate Water Stress | 0 | 0 |
| Severe Water Stress | 0 | 0 |
| Disease Confounded | 7 | 0 |

Result: **insufficient classes/groups for a credible generalization estimate**. No folds or model fits were created for the real grouped dataset. There are no new valid performance estimates and no replacement production model. The existing “Model update required” outcome remains appropriate.

The optional five-model [legacy reproduction](../reports/legacy_development_only_2026-10-08_191828_3ae0f250.md) used the original 40 feature rows. Its numbers are image-level, not group-independent, not gate-checked and **must not be reported as performance**. Synthetic grouped data exercises the successful nested evaluation path in tests; those results are not project performance evidence.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider`: **87 passed**.
- `node tests/frontend_stage1.cjs`: **passed**, including the new button label and existing upload/error/reset/preview behavior.
- Tests cover manifest completeness and historical gate agreement, duplicate conflicts, identity resolution, tampered gate annotations, import safety, artifact non-overwrite, small-class rejection, group/hash separation in inner and outer folds, deterministic selection, pipeline scaling, error metrics, distinct report names and historical diagnostic reproduction.
- Existing preservation tests confirm the protected originals match their backup hashes. Historical Stage 3 reports and threshold configuration are unchanged.
- Installed NumPy/joblib compatibility and SVC probability deprecation warnings remain. They do not fail this run; dependency pinning and calibration work remain later-stage scope.

## What the owner must supply

Fill `plant_id`, `leaf_id`, `session_id`, `source`, `device`, `background_type`, `label_method` and relevant notes from records. Unknown values should stay blank. Collect gate-passing Moderate and Severe examples and enough independently documented plants across all classes, following the collection protocol. Do not relax gates just to recover class counts. A final untouched set of independent plants and later sessions still needs collection.

Stage 5 has not started.
