# Lecor implementation report — Stages 1–6

2026-10-09. **No validated classifier exists yet. Classification results cannot be presented as performance.** The application is ready to demonstrate image-processing safeguards in explicit preview-only mode, not validated water-stress classification. All six authorized implementation stages have been addressed; biological data collection, real-device acceptance and model validation remain outstanding.

## File-by-file implementation changes

| File / artifact | Final change |
|---|---|
| `app.py` | UUID temporary uploads; content/pixel/decode validation; cleanup and JSON errors; safe startup; versioned model contract; no predictions for rejected/uncertain/incompatible inputs; faithful LR explanation; explicit preview-only startup and responses; preview withholds old benchmark numbers |
| `config.py` | Project-relative paths; upload/server settings; named fixed-resolution/gate constants; preprocessing signature; provisional confidence threshold; preview environment flag |
| `feature_extraction.py` | Empty-region visualization fix; shared resize/segmentation/gate path; border diagnostic; preserved 15-feature math; corrected visual/shape/score/advisory claims; ablation-only chromaticity/LAB/oriented shape function |
| `templates/index.html` | Clear old results, lock Reset, revoke object URLs, readable network errors; corrected labels/limitations; masks and retake feedback; model-update wording; LR contributions separated from dataset comparisons; preview banner/pipeline; keyboard upload semantics and all-tab print CSS |
| `dataset_manifest.py` | Reproducible image inventory, hashes, dimensions, preserved human blanks, duplicate conflicts, conservative plant/session grouping |
| `dataset_manifest.csv` | 56 file rows, 40 unique hashes, duplicate links and gate diagnostics; no invented plant/leaf/session/source/label metadata |
| `dataset_manifest.metadata.json` | Initial inventory provenance, unchanged gate/preprocessing snapshot and diagnostic definitions |
| `build_dataset.py` | Import-safe, manifest/hash validation, conflicts, recomputed gate, missing-class refusal, versioned feature CSV and schema/provenance sidecar |
| `train_model.py` | Historical image-level methodology explicitly labelled; fixed production feature order; compatible input metadata required; exclusive versioned bundle/benchmark writes with schema, dataset hash, runtime versions, date and actual CV protocol |
| `evaluate_grouped.py` | Readiness guard; plant/session groups; duplicate/group isolation; nested grouped LR/SVM/regularized RF comparison; optional baselines; full-precision tie-break; class/error metrics; separate warned legacy mode |
| `ablate_features.py` | Five exploratory variants only inside nested grouped CV, shared partitions, synthetic execution proof, real-data blocked reports, no production selection or model save |
| `model_contract.py` | Strict bundle and estimator schema/version/runtime validation; exact predicted-class Logistic Regression score decomposition |
| `report_mask_quality.py` | Unique-image mask report, aliases, all reason codes, overlapping failure counts; no deletion of images |
| `timing_preview.py` | Real local in-process upload/processing response measurements, p50/p95 and per-image outcomes; expressly not classifier latency |
| `record_test_run.py` | Actual pytest XML/log capture, dependency inventory, frontend check and pip-check evidence |
| `generate_test_results.py` | Per-case Markdown outcomes from recorded JUnit evidence, with no invented passes |
| `requirements.txt` | Exact runtime pins; NumPy 2.4.6/joblib 1.5.3 verified free of original-loading warnings |
| `requirements-dev.txt` | Runtime requirements plus pinned pytest 9.1.1 |
| `requirements-py313.lock.txt` | Exact direct/transitive closure of clean Python 3.13.0 test environment |
| `requirements-py314.lock.txt` | Exact relevant dependency closure of Python 3.14.5 test environment |
| `.gitignore` | Temporary uploads, Python caches and local virtual environments excluded |
| `tests/test_stage1.py` | Upload/security/error/cleanup/original-preservation regressions and safe startup |
| `tests/test_stage2.py` | Corrected claims, score suppression, cautious guidance and descriptive comparisons |
| `tests/test_stage3.py` | Standardization, masks, thresholds, rejects, uncertainty and incompatible-model behavior |
| `tests/test_stage4.py` | Manifest/group/conflict/leakage/readiness/selection/metrics checks and legacy reproduction |
| `tests/test_stage5.py` | Model provenance round-trip, mismatch refusal, LR math/API, ablation features/folds, warning-free legacy-byte loading |
| `tests/test_stage6.py` | Preview-only startup/no-inference, synthetic contour/color/tiny-image checks, acquisition failures, truncation and repeated-image consistency |
| `tests/conftest.py` | Explicit test-only legacy fixtures for earlier regression paths; production loader still refuses original artifact |
| `tests/frontend_stage1.cjs` | Executes actual inline JavaScript in a minimal DOM, covering upload/error/reset/labels/explanations/preview behavior |
| `backup_original/` | Protected original copies and SHA-256 manifest, checked by tests; not rewritten |
| `reports/mask_quality_report.csv`, `mask_quality_summary.md` | Historical Stage 3 per-image gate evidence, all reason codes and counts; preserved |
| `reports/grouped_eval_*.json/.md` | Real gate-passing readiness refusal with class/group deficits; no fabricated folds |
| `reports/legacy_development_only_*.json/.md` | Historical 40-row image-level diagnostics, clearly not performance |
| `reports/ablation_*.json/.md` | Real blocked result and separately labelled synthetic grouped execution evidence |
| `reports/preview_timing_*.json` | Actual preview response timings and limitations |
| `reports/test_run_*.json/.xml/.log` | Actual environment and suite evidence for both Python versions |
| `docs/stage1_report.md` through `stage6_report.md` | Per-stage scope, observed checks and remaining work; earlier stages are historical snapshots |
| `docs/srs_change_suggestions.md` | Exact proposed requirement/limitation/version edits; PDF unchanged |
| `docs/data_collection_protocol.md` | Practical acquisition, labels, suggested per-class targets, collector table and untouched-test policy |
| `docs/dataset_manifest_guide.md` | Computed/human fields, gate diagnostics, identity grouping and preservation rules |
| `docs/environment.md` | Exact tested versions, environment construction and recreation commands |
| `docs/test_results.md` | Generated actual per-test pass/fail evidence; scientific limitations explicit |
| `docs/requirements_traceability.md` | REQ-001–REQ-016 code/test/status mapping and course-service caveats |
| `docs/SDD.md` | Architecture, data flow, module boundaries, persistence and design decisions |
| `docs/STD.md` | Test strategy, executed evidence, limitations and future real-world evaluation protocol |
| `docs/limitations.md` | Dataset, label, background, physiological, single-view shape, evaluation and deployment limitations |
| `docs/manual_verification_checklist.md` | Phone/laptop camera, browsers, keyboard, printing and acceptance log |
| `docs/final_report.md` | This consolidated implementation and remaining-work report |

## Test outcomes and environment

- CPython **3.14.5**: **134 passed**, no failures/skips; relevant packages inherited except local NumPy pin.
- CPython **3.13.0**: **134 passed**, no failures/skips in a clean virtual environment.
- Both: macOS 26.5.1 arm64; frontend checks with Node 24.15.0 and `pip check` passed.
- Both: original artifact deserialization emitted no warnings under NumPy 2.4.6/joblib 1.5.3. Eleven legacy SVC probability-training deprecation warnings remain per suite.
- Real localhost preview server and multipart upload smoke check passed on Python 3.13.0; server stopped after checking.
- Preview response timing over 120 requests: p50 **0.0127 s**, p95 **0.1050 s**. This excludes actual classification and browser/network. REQ-013 remains unverified for classification.

See `test_results.md` and `environment.md` for raw evidence, every case and all dependency versions. Camera, printing and cross-browser behavior remain **needs manual verification on device**.

## Evaluation numbers and what they mean

| Evidence | Unit/protocol | Result | Permitted interpretation |
|---|---|---|---|
| Real grouped evaluation | Intended plant/session grouped nested CV | **No estimate produced**: Healthy 4, Moderate 0, Severe 0, Disease 7 accepted unique images; no documented groups | Insufficient classes/groups for a credible generalization estimate |
| Real feature ablation | Intended grouped nested CV | **No estimate produced** | Blocked before fitting; no production feature selection |
| Original 40-row LR diagnostic reproduction | Image-level StratifiedKFold, not group-independent or gate-checked | Accuracy 0.6250, Macro F1 approximately 0.6019 | Historical development diagnostics only; **must not be reported as performance** |
| Synthetic ablation execution | Invented pixels/labels, six invented groups, nested CV | All five variants ran; trivially separable fixture Macro F1 1.0 | Software execution evidence only; **not project accuracy** or a reason to change production features |

No new production classifier was selected. The existing 15 production features and original model are preserved. Original image-level results do not become valid because software tests pass.

## Still depends on the owner

- Fill verified `plant_id`, `leaf_id`, `session_id`, `source`, `device`, `background_type`, `label_method` and notes. Leave unknown values blank; do not infer them from filenames.
- Collect suitable accepted Moderate/Severe examples and sufficiently independent plants/sessions across all classes, with defensible labels. Suggested collection targets are planning aids, not statistically derived sample sizes.
- Reserve entirely new plants in later sessions for a final test touched once after the methodology is frozen.
- Review ablations only after real grouped readiness; approve any production feature change and future compatible model explicitly.
- Complete real-device camera, laptop webcam, printing, browser and keyboard checks. No explicit webcam integration currently exists.
- Approve/apply SRS wording and version corrections; confirm the authoritative document version. The PDF was not edited.
- Obtain evaluator agreement that the actually implemented multimedia/application services meet the course minimum. Do not assume temporary storage or an unverified camera counts.
- Collect independent physiological measurements before claiming quantitative water-stress validity, disease discrimination or early-warning effectiveness.

## What to show in the demo and what to say

Run `LECOR_PREVIEW_ONLY=true python app.py` in the documented environment. Show `healthy_006.JPG` as an image-processing example, the selected/candidate masks and Pipeline tab. Then show a clipped/blank or uniform-green input receiving rejection reasons without a class/score. Demonstrate a readable invalid-upload error. Show the manifest, blocked real grouped report and test evidence.

Say: **“This prototype demonstrates image processing and quality safeguards. No validated classifier exists yet. Independent labels and documented plant/session groups are needed before we can report water-stress classification performance.”**

Do not call a passing mask a healthy diagnosis, a green heatmap water content, a synthetic result project accuracy, or the retained heuristic a physiological index. Explain that the tests verify software and that real-device checks remain open.
