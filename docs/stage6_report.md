# Stage 6 — Tests, documentation and preview-only demonstration

Completed 2026-10-09. **No validated classifier exists yet. Classification results cannot be presented as performance.** Production features/model, protected originals and gate thresholds remain unchanged.

## Changes

- Explicit `LECOR_PREVIEW_ONLY=true` startup skips model loading and disables classification. A visible notice explains the limitation. Passing inputs show masks and segmented/color/GLI previews; rejected inputs show reasons. Historical benchmark numbers are withheld in this mode.
- Added 27 parametrized/software test cases, extending the suite from 107 to 134. Coverage includes preview startup without a model, no model calls even with an estimator present, repeatability, contour/color math, tiny visualizations, acquisition failures, truncated files and schema mismatches.
- Added actual-run recorder, Markdown evidence generator and preview p50/p95 timing script.
- Added SDD, STD, requirements traceability, limitations, exact environment instructions, dependency closure locks and a manual device checklist.
- Updated README and current SRS suggestion status notes. The original PDF is unchanged.
- Added keyboard activation semantics to the upload control and print CSS that exposes all tabs. These still require device verification.

## Actual checks

| Check | Result |
|---|---|
| CPython 3.14.5, macOS arm64 | 134 passed, 0 failed, 0 skipped |
| CPython 3.13.0, clean macOS arm64 environment | 134 passed, 0 failed, 0 skipped |
| Frontend DOM regression, Node 24.15.0 | Passed in both recorded runs |
| Dependency consistency (`pip check`) | Passed in both environments |
| Original model deserialization with NumPy 2.4.6/joblib 1.5.3 | Zero warnings, tested in both environments; original remains contract-incompatible |
| Localhost HTTP smoke check on Python 3.13.0 | Real server home banner and valid multipart upload passed; response was preview-only with null class/score; server stopped afterward |
| Preview timing | 40 unique images × 3 repeats; p50 0.0127 s, p95 0.1050 s |

Both suites produced 11 remaining SVC probability-training deprecation warnings. These are recorded, not hidden. No browser/device or physiological validation was performed. Preview timings exclude browser/network/classifier and do not verify REQ-013 classification latency. The initial localhost check was blocked by the execution sandbox; the authorized check outside that restriction passed. That was an environment permission issue, not an application defect.

Raw evidence and every test outcome are linked from `test_results.md`. Exact package versions and recreation commands are in `environment.md` and the two lock files.

## What to show in the demo and what to say

1. Start with `LECOR_PREVIEW_ONLY=true python app.py` in the recreated environment. Point to the visible no-classifier notice.
2. Upload `dataset/Healthy/healthy_006.JPG`. Show selected/candidate masks and the Pipeline tab. Say: “This is a segmentation and color visualization, not a healthy/stressed diagnosis.”
3. Upload a clipped/uniform-background example or blank image. Show the quality reasons and absence of a class/score. Show readable error handling with a non-image file if desired.
4. Show the manifest gaps, collection protocol and blocked grouped-evaluation report. Say: “No validated classifier exists yet. Independent labels and documented plant/session groups are still required before classification performance can be reported.”
5. Present automated test evidence as software verification only. Do not show legacy CV or synthetic ablation scores as project performance. Explain that camera capture, printing and cross-browser checks still need real-device verification.

## Remaining owner work

Fill verified identity/provenance fields, collect accepted Moderate/Severe samples and independent plants across all classes, reserve an untouched final set, complete the manual device checklist, confirm course-service acceptance and approve/apply SRS wording/version edits. Production feature changes and a future compatible model still require explicit review. All six requested implementation stages are complete; these data-dependent/manual validation tasks remain open.
