# Whole-plant development diagnostics

EXPERIMENTAL DEVELOPMENT ONLY. Folder labels are unverified. Image-level CV is not plant/session-independent; near-duplicates and background/source confounding may inflate scores. Model selection uses the same CV, so the selected score is optimistic. No untouched test set or validated classifier exists. Do not present these numbers as real-world performance.

55 files; 54 unique gate-passing images; counts: {'Healthy': 14, 'Severe_Water_Stress': 14, 'Moderate_Water_Stress': 13, 'Disease_Confounded': 13}.

| Candidate | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.426 | 0.429 | 0.430 |
| RBF SVM | 0.463 | 0.429 | 0.466 |
| Random Forest | 0.426 | 0.427 | 0.429 |

Selected development model: Logistic Regression. No independent test estimate.

Near-duplicate candidates (not proof): []

Full per-class metrics, confusion matrices and out-of-fold predictions: development_report.json.

Fill plant_id, session_id, source, label_method, device and background_type in manifest. IDs must describe real identities, not invented row numbers.
