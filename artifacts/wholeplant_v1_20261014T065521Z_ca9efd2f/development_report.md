# Whole-plant development diagnostics

EXPERIMENTAL DEVELOPMENT ONLY. Folder labels are unverified. Image-level CV is not plant/session-independent; near-duplicates and background/source confounding may inflate scores. Model selection uses the same CV, so the selected score is optimistic. No untouched test set or validated classifier exists. Do not present these numbers as real-world performance.

55 files; 55 unique gate-passing images; counts: {'Healthy': 15, 'Severe_Water_Stress': 14, 'Moderate_Water_Stress': 13, 'Disease_Confounded': 13}.

| Candidate | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.527 | 0.534 | 0.525 |
| RBF SVM | 0.491 | 0.443 | 0.491 |
| Random Forest | 0.473 | 0.461 | 0.473 |

Selected development model: Logistic Regression. No independent test estimate.

Near-duplicate candidates (not proof): [{"first": "Severe_Water_Stress/severe_013.JPG", "second": "Severe_Water_Stress/severe_014.JPG", "dhash_distance": 8, "status": "candidate only; manual review required"}]

Full per-class metrics, confusion matrices and out-of-fold predictions: development_report.json.

Fill plant_id, session_id, source, label_method, device and background_type in manifest. IDs must describe real identities, not invented row numbers.
