# Software Test Document — Lecor

Version: Stage 6, 2026-10-09.

**No validated classifier exists yet. Classification results cannot be presented as performance.** Automated tests prove specified software behavior on fixtures; they do not prove drought/disease discrimination, calibrated stress severity or field generalization.

## Test strategy

1. Unit checks: contour formulas on rectangle/circle/degenerate shapes, RGB channel order, GLI bounds, ExG identity, experimental feature math, finite outputs and empty-mask handling.
2. Image integration: real healthy regressions, zero yellow/dark overlays, tiny images, blank/uniform/low-light/bright/complex inputs, clipped/multiple candidates and corrupt/truncated bytes. A non-leaf green object passing the gate is deliberately documented as a limitation.
3. Model contract: missing/extra/reordered features, version/library mismatch, class alignment, prediction/probability shapes, provenance writer and load refusal. Synthetic fitted estimators exercise valid paths in temporary files only.
4. Evaluation: no group/hash overlap in inner or outer partitions, full-precision tie-break, scaler pipelines, same grouped partitions across feature variants, and refusal before fitting when real classes/groups are missing.
5. Flask: UUID paths, request/pixel limits, validation, cleanup on error, structured JSON, missing model behavior, preview-only startup and complete classification suppression even with a model in memory.
6. Frontend: actual inline JavaScript in a minimal DOM tests stale clearing, Reset lock, object URL revocation, non-JSON errors, descriptive labels, coefficient rendering, mask feedback and preview-only pipeline images. This is not a real browser/device test.
7. Reproducibility: repeated real-image extraction/visualizations, identical preview responses and deterministic synthetic nested evaluation.
8. Timing: three sequential repetitions of all 40 byte-unique images through Flask's in-process client, one warm-up excluded. Measures preview/rejection processing only.

## Executed evidence

`test_results.md` is generated from actual pytest JUnit XML and environment records and lists each test's status per Python version. Raw XML/log/JSON evidence is under `reports/test_run_*`. The report also records frontend tests and `pip check`. `environment.md` describes exact versions and recreation commands. Do not replace observed outcomes with expected passes.

Preview timing is in `reports/preview_timing_2026-10-09_113614_4e884abc.json`: 120 requests, p50 approximately 0.0127 seconds, p95 approximately 0.1050 seconds on the recorded machine. These timings exclude a real server socket, browser/network, startup/model loading and classification, so **REQ-013's classification-within-10-seconds claim remains not verified**.

## Known failures, limitations and warnings

- Scientific readiness is blocked: no accepted Moderate/Severe examples and no documented plant/session groups. Real grouped evaluation and ablation intentionally return the insufficient-classes/groups status; these are correct refusals, not successful performance measurements.
- The original classifier lacks compatible preprocessing/schema metadata; normal startup refuses it. Explicit preview mode works without it.
- A centered green non-leaf object may pass. Uniform green background, clipping and poor light can invalidate segmentation. The gate is provisional, not semantic leaf recognition.
- Existing SVC probability-training deprecation warnings remain. The pinned NumPy/joblib pair loads original bytes without those earlier deserialization warnings. No warning is suppressed by application code.
- Camera (REQ-001), printing and cross-browser behavior: **needs manual verification on device**. There is no explicit laptop webcam capture path.
- Print CSS exposes all tabs, but real pagination, image visibility and classified-result completeness are not established by DOM tests. Preview mode cannot demonstrate unavailable classification result sections.
- Automated suite failures, if any, are listed in the generated results; do not describe scientific limitations as fixed merely because tests pass.

## Real-world evaluation protocol

Follow `data_collection_protocol.md`. Record stable plant/leaf/session/device identities and independent labeling evidence. Use single whole leaves, neutral background and diffuse light, with framing represented across every class. Retain rejection coverage and reason counts alongside any future metrics.

After resolving duplicate conflicts and missing metadata, pre-register grouping, feature candidates and selection metric. Use grouped nested CV for development only when every partition is feasible. Reserve entirely new plants acquired in later sessions as a final untouched test set. Freeze extraction, thresholds, model, calibration and interpretation before examining that set. Evaluate it once and report all class metrics, confusion matrix, severe-to-healthy misses, disease-to-stress confusion and healthy false-alert rate. Also report rejection and uncertainty coverage. Subsequent tuning requires a new independent test set. Physiological score validation needs separately measured water status; RGB class confidence is not that measurement.

## Manual acceptance

Complete `manual_verification_checklist.md` on real devices. Record browser/OS/version, observed behavior and evidence rather than ticking inferred passes. Course evaluator confirmation of the minimum service categories remains outstanding.
