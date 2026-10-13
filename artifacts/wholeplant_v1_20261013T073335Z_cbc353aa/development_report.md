# Whole-plant development diagnostics

EXPERIMENTAL DEVELOPMENT ONLY. Folder labels are unverified. Image-level CV is not plant/session-independent; near-duplicates and background/source confounding may inflate scores. Model selection uses the same CV, so the selected score is optimistic. No untouched test set or validated classifier exists. Do not present these numbers as real-world performance.

44 files; 44 unique gate-passing images; counts: {'Healthy': 11, 'Moderate_Water_Stress': 11, 'Severe_Water_Stress': 11, 'Disease_Confounded': 11}.

| Candidate | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.432 | 0.425 | 0.432 |
| RBF SVM | 0.409 | 0.389 | 0.409 |
| Random Forest | 0.455 | 0.432 | 0.455 |

Selected development model: Random Forest. No independent test estimate.

Near-duplicate candidates (not proof): []

Full per-class metrics, confusion matrices and out-of-fold predictions: development_report.json.

Fill plant_id, session_id, source, label_method, device and background_type in manifest. IDs must describe real identities, not invented row numbers.
