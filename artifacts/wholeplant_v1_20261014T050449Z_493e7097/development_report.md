# Whole-plant development diagnostics

EXPERIMENTAL DEVELOPMENT ONLY. Folder labels are unverified. Image-level CV is not plant/session-independent; near-duplicates and background/source confounding may inflate scores. Model selection uses the same CV, so the selected score is optimistic. No untouched test set or validated classifier exists. Do not present these numbers as real-world performance.

55 files; 55 unique gate-passing images; counts: {'Healthy': 15, 'Severe_Water_Stress': 14, 'Moderate_Water_Stress': 13, 'Disease_Confounded': 13}.

| Candidate | Accuracy | Macro F1 | Balanced accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.345 | 0.317 | 0.336 |
| RBF SVM | 0.400 | 0.335 | 0.392 |
| Random Forest | 0.364 | 0.351 | 0.352 |

Selected development model: Random Forest. No independent test estimate.

Near-duplicate candidates (not proof): [{"first": "Healthy/helathy_013.JPG", "second": "Moderate_Water_Stress/moderate_004.JPG", "dhash_distance": 8, "status": "candidate only; manual review required"}, {"first": "Healthy/helathy_013.JPG", "second": "Severe_Water_Stress/severe_007.JPG", "dhash_distance": 8, "status": "candidate only; manual review required"}, {"first": "Healthy/helathy_013.JPG", "second": "Disease_Confounded/IMG_4846.JPG", "dhash_distance": 5, "status": "candidate only; manual review required"}, {"first": "Moderate_Water_Stress/moderate_004.JPG", "second": "Severe_Water_Stress/severe_001.JPG", "dhash_distance": 5, "status": "candidate only; manual review required"}, {"first": "Moderate_Water_Stress/moderate_004.JPG", "second": "Disease_Confounded/IMG_4846.JPG", "dhash_distance": 7, "status": "candidate only; manual review required"}, {"first": "Severe_Water_Stress/severe_005.JPG", "second": "Disease_Confounded/IMG_4846.JPG", "dhash_distance": 8, "status": "candidate only; manual review required"}, {"first": "Severe_Water_Stress/severe_009.JPG", "second": "Disease_Confounded/IMG_4846.JPG", "dhash_distance": 8, "status": "candidate only; manual review required"}]

Full per-class metrics, confusion matrices and out-of-fold predictions: development_report.json.

Fill plant_id, session_id, source, label_method, device and background_type in manifest. IDs must describe real identities, not invented row numbers.
