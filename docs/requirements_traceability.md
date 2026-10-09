# Requirements traceability

Status as of 2026-10-09. **No validated classifier exists yet. Classification results cannot be presented as performance.** Software tests using synthetic or legacy fixtures verify code paths, not agronomic validity. No compatible production classifier is active.

Source: `Lecor_SRS_v0.3-4.pdf`, pages 8–9, read from the Downloads copy because the originally supplied WhatsApp temporary path no longer exists. The cover/version inconsistency and proposed wording corrections remain in `srs_change_suggestions.md`. The PDF was not changed. Summaries below preserve the original requirements; they do not silently replace physiological claims with validated ones.

| Requirement | Implementing files/functions | Automated evidence | Status and outstanding work |
|---|---|---|---|
| REQ-001: camera or uploaded image | `templates/index.html` file input; `app.predict`, `validate_image` | `test_stage1.py` upload/format/size tests; frontend file selection | **partial — needs manual verification on device** for camera. Upload works. No explicit laptop webcam/getUserMedia implementation; phone file-picker camera availability depends on device. |
| REQ-002: class, score and explanation in UI | `app.predict`; indicator/advisory; `renderOutput`; model contract | Stage 1/2 API fixtures, Stage 5 explanation API test | **partial**. Presentation paths tested with fixtures; no validated classifier or currently available classification. Score is an unvalidated heuristic, not physiological stress. |
| REQ-003: processing images | `generate_visual_pipeline`, `analyze_leaf_full`; `setPipelineImages`, preview masks | Healthy visualization, tiny-image tests; frontend preview/pipeline test | **complete** for preview visualization implementation. Maps are heuristic pixel colors, not tissue-health diagnoses; semantic segmentation reliability unverified. |
| REQ-004: HSV/RGB and vegetation indices | `extract_color_features` | `test_rgb_order_gli_bounds_exg_identity_and_empty_mask` | **complete** for numerical extraction. Physiological relevance is not validated. |
| REQ-005: outline and shape/curl indicators | `_segment_candidates`, `extract_shape_features` | Synthetic rectangle/circle/degenerate contour tests, clipping/multiple-component tests | **partial**. Shape math implemented; outline is a provisional candidate. No direct curl/turgor measurement. |
| REQ-006: four-class trained classification | `app.predict`, `validate_bundle` | Shape/class/probability contract tests and synthetic API test | **partial**. No validated classifier exists; accepted Moderate/Severe data and documented groups missing. |
| REQ-007: compare/select candidate models with CV | `evaluate_nested`, `selection_key`; historical `run_benchmarks` | Stage 4 nested-group/no-leakage/tie-break tests; Stage 5 synthetic ablations | **partial**. Grouped pipeline works on synthetic fixtures; real evaluation blocks. Historical image-level selection is not performance evidence. No new production winner selected. |
| REQ-008: invalid image/no-leaf errors | `validate_image`, `check_mask_quality`, JSON handlers; `executeScan` | Bad/truncated/oversized upload, blank/complex/clipped/multiple inputs, error cleanup | **partial**. Structured validation/rejections implemented; a green non-leaf object may pass. Successful mask is not proof of a leaf. |
| REQ-009: 0–100 stress score | `compute_visual_stress_indicator`, UI integer display | Stage 2 integer/bounds/disease suppression; preview no-score tests | **partial**. Heuristic implementation only, suppressed in preview/rejected/uncertain/confounded outcomes. No physiological calibration. |
| REQ-010: plain-language comparison explanation | `compare_feature_measurements`, `explain_logistic`, advisory; two distinct panels | Stage 2 descriptive comparisons; Stage 5 exact decision-score reconstruction | **partial**. Code verified; live classification blocked. Pooled averages are not healthy norms. Model contributions are not biological causes. |
| REQ-011: simple first-time workflow | Upload/scan/reset, visible preview notice, JSON/fetch handling | Frontend request/reset/stale-state checks | **not verified** for first-time users. Conduct observed usability sessions and keyboard/device checks. |
| REQ-012: colors, bars and probabilities | `renderOutput`, textual labels; conservative hidden result sections | Stage 2 frontend formatting; preview never shows classification | **partial**. Fixture-based rendering only; accessibility and browser behavior need manual verification. |
| REQ-013: classification within 10 seconds on ordinary laptop | `timing_preview.measure`; inference route | Actual 120-request preview timing report; response/schema tests | **not verified** for classification. Preview p50/p95 exclude browser, network and classifier; cannot establish this requirement. |
| REQ-014: OpenCV image processing | Shared `feature_extraction.py` and cv2 decoding | Feature/segmentation/visualization suite | **complete** for CV operations. Pillow additionally validates image format/dimensions before cv2 decoding. |
| REQ-015: scikit-learn classifiers | `train_model.py`, `evaluate_grouped.py`, `model_contract.py` | Pipeline, saved-bundle, nested-CV and synthetic writer tests | **complete** as a technology constraint; this does not validate classification performance. |
| REQ-016: Flask backend | `app.py`: `/`, `/predict`, `/api/benchmark` | Flask client tests; preview startup subprocess tests | **complete** as a technology constraint. Hosted production deployment not evaluated. |

## Additional acceptance items

| Item | Evidence/status |
|---|---|
| Camera (REQ-001) | **needs manual verification on device**. Phone picker and laptop webcam separately; lack of explicit webcam capture must be reported, not marked passed. |
| Printing/PDF completeness | **needs manual verification on device**. Buttons invoke browser print; print CSS exposes all tabs. Check mask/pipeline/limitations and, later, every classified-result section. Automated DOM tests do not prove pagination or visibility. |
| Chrome/Firefox/Safari/Edge | **needs manual verification on device**. No cross-browser certification claimed. |
| Keyboard navigation | Upload control now exposes a keyboard role/tab stop and Enter/Space activation; real focus/file-picker behavior remains manual. |
| Repeated-input consistency | Exact features/visualizations and preview responses tested; synthetic classifier fixtures do not establish a reliable real-world classifier. |
| Privacy/security | UUID temporary uploads, cleanup, size limits, decoded-image checks, safe local defaults tested. HTTPS hosting remains deployment work; no database/history exists. |

## Course alignment

OpenCV is an explicitly permitted tool in the supplied course sheet. Implemented evidence includes image effects/visualization, UI design and temporary file storage. Camera is unverified, and there is no audio, synchronization, animation workflow or database. Whether temporary storage and these categories count as the course's minimum three distinct services requires evaluator confirmation; do not claim that minimum is already accepted. SRS suggestions, SDD, STD, test evidence and the manual checklist provide academic documentation, not marks or approval. Record final course acceptance separately.
