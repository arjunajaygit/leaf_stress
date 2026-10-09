# Limitations and scientific status

**No validated classifier exists yet. Classification results cannot be presented as performance.** Preview-only mode demonstrates software and image processing, not stress detection accuracy.

- Dataset size: 56 files represent 40 distinct image hashes, not 40 proven independent plants. There are 16 duplicate copies. The unchanged gate accepts only 11 unique images (Healthy 4, Disease Confounded 7). No Moderate or Severe images pass.
- Identity and labels: plant, leaf and session IDs, source and label-method provenance remain blank. Independent biological ground truth and disease diversity are unverified. Do not infer metadata from filenames or appearance.
- Confounding: class may correlate with background, camera, session, illumination or framing. Different views of the same leaf/plant may not be independent. Byte deduplication does not establish plant independence or catch all near duplicates.
- Segmentation: fixed HSV rules and morphology are provisional. They can merge green backgrounds, exclude discolored tissue, fail under shadow/glare, reject valid leaves or accept green non-leaf objects. Gate thresholds and the confidence cutoff are not validated operating points.
- Visual-only: RGB/HSV and RGB indices are not calibrated reflectance, water content or physiological measurements. Drought, disease, nutrient problems, pests and heat may look similar or coexist. Single-image early warning is unvalidated.
- Shape: solidity, circularity, extent and aspect ratio are single-view, 2D projected descriptors. Cultivar, leaf maturity, natural shape, perspective and clipping confound them. They do not directly measure curl, leaf angle, turgor or water content.
- Score: the retained 0–100 visual indicator is arbitrary and uncalibrated against physiology. Probability is not stress severity; calibration remains unverified. No score/class is shown in preview mode, on rejected/uncertain inputs or with an incompatible model; the indicator is also suppressed for the confounded class.
- Explanation: pooled dataset deviations are descriptive. Logistic Regression terms reconstruct a model decision score, not causal evidence or a probability decomposition; correlated features can redistribute coefficients.
- Evaluation: historical image-level CV reused data for selection, lacks known independent groups and is not gate-checked. It must not be presented as performance. Synthetic grouped ablation scores prove execution only. Real grouped evaluation cannot proceed yet.
- Deployment: no compatible production model has been activated. Exact model-library/Python-version checks may require the recorded environment or retraining. Successful pickle loading does not imply methodological validity; only trusted artifacts may be loaded.
- Interface: REQ-001 camera, printing and cross-browser behavior **need manual verification on device**. No explicit laptop webcam integration exists. DOM checks cannot prove screen-reader behavior, browser printing or mobile capture.
- Timing: current response measurements are local preview processing with no browser/network/classifier. They do not verify REQ-013 classification latency.
- Scope: no scan-history database, longitudinal monitoring, audio/video synchronization, laboratory replacement, treatment prescription or accepted course-service count is claimed.

See the collection protocol and manual checklist for the evidence still required. Keep all rejected images and provenance records; do not relax thresholds just to produce four-class counts or inflate results.
