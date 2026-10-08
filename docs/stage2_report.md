# Stage 2 — Correct the claims

Date: 2026-10-08 (Asia/Kolkata). Branch: `audit-fixes`.

## Changes

- Replaced the old score name with **Experimental relative visual-stress indicator**, an integer 0–100 heuristic without a percentage sign. The formula remains unvalidated. Disease-confounded predictions return `null` and do not display the indicator.
- Replaced the attribution panel with **Feature measurements vs. dataset averages**. It presents all 15 available measurements in schema order with pooled all-class means and z-scores, without Random Forest importance weighting or biological interpretations. These comparisons are not prediction contributions. Model-specific explanations remain Stage 5 work.
- Renamed the comparison panel to **Predictions from other models on this image** and excluded the already-displayed primary model from that panel.
- Centralized frontend disease-label presentation: **Disease-like or other confounding symptoms suspected** appears in the result, probability bars and other-model predictions. Saved class identifiers and artifacts are unchanged.
- Replaced prescriptive advisory text with inspection, direct soil-moisture checking, consideration of alternative causes and expert consultation. Removed exact irrigation quantities, recovery times and fungicide instructions.
- Added visible result limitations and corrected projected-shape, color-map, segmentation and probability wording. Color-group percentages remain percentages of masked pixels, not physiological severity or confirmed tissue diagnoses.
- Rewrote README around actual current behavior and limitations, including image-level development CV, unsupported independence, unchanged heuristic weights, versioned artifacts and pending device verification.
- Added `docs/srs_change_suggestions.md` with current and proposed wording for REQ-002/005/008/009/010, section 1.3.4, and the cover/filename version discrepancy. It explicitly resolves the REQ-008 versus outdated validation limitation ambiguity. The source PDF was not edited; the owner must confirm the authoritative version and apply approved changes.

API changes: `cwsi` is replaced by `visual_stress_indicator`; `xai_attributions` by `feature_comparisons`; advisory fields are now `guidance` and `action_items`. The bundled frontend is updated together. No compatibility aliases preserve the retired scientific claims. Legacy color-map internal keys are documented in README.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node tests/frontend_stage1.cjs
git diff --check
```

- **42 Python tests passed** in the actual run, including Stage 1 regressions and Stage 2 claim/output contracts.
- Frontend logic checks passed integer rendering, disease suppression, reappearance on subsequent non-disease results, all disease display labels, dataset comparisons and the Stage 1 request/error lifecycle.
- All 40 saved feature rows still match the shared extractor. Production features, segmentation and saved models were not changed.
- Original protected files and backup copies still match their SHA-256 manifest.
- Existing joblib/NumPy and SVM probability deprecation warnings remain (1,043 warnings in the Python run). They are not new scientific results or test failures.
- JavaScript tests use a minimal DOM; physical device, layout, printing and cross-browser checks remain pending.

## Remaining

Stop after this stage. Stage 3 mask-quality rejection and uncertainty, Stage 4 provenance/grouped evaluation, Stage 5 ablations/reproducibility/model explanation, and Stage 6 expanded documentation/testing have not been implemented. The revised language does not make the present classifier or indicator scientifically validated. No new independent evaluation numbers are claimed.
