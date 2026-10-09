# LEGACY DEVELOPMENT-ONLY — NOT PERFORMANCE

**LEGACY DEVELOPMENT-ONLY: image-level numbers are NOT group-independent, NOT gate-checked, and MUST NOT be reported as performance. They are historical diagnostic reproductions only. The original features are incompatible with current resized preprocessing.**

- LEGACY DEVELOPMENT-ONLY: image-level numbers are NOT group-independent, NOT gate-checked, and MUST NOT be reported as performance. They are historical diagnostic reproductions only. The original features are incompatible with current resized preprocessing.
- No tuning or production-model selection is performed. Original candidate settings are used for historical reproduction.

Generated: 2026-10-08T19:18:28.978438+05:30

Groups per class: unknown. Gate status: not checked. Original 40 feature rows used unchanged.

### Logistic Regression (L2) — development-only diagnostics

| Metric | Value |
|---|---:|
| accuracy | 0.625 |
| macro_f1 | 0.6019464421924375 |
| balanced_accuracy | 0.6000631313131314 |
| macro_precision | 0.6118881118881119 |
| macro_recall | 0.6000631313131314 |
| weighted_f1 | 0.6254215702299226 |
| severe_to_healthy | 1 |
| disease_to_water_stress | 1 |
| healthy_false_alert_rate | 0.625 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.5 | 0.375 | 0.42857142857142855 | 8.0 |
| Moderate_Water_Stress | 0.5 | 0.5555555555555556 | 0.5263157894736842 | 9.0 |
| Severe_Water_Stress | 0.5384615384615384 | 0.6363636363636364 | 0.5833333333333334 | 11.0 |
| Disease_Confounded | 0.9090909090909091 | 0.8333333333333334 | 0.8695652173913043 | 12.0 |

Confusion matrix (rows actual, columns predicted; order below):

```json
{
  "labels": [
    "Healthy",
    "Moderate_Water_Stress",
    "Severe_Water_Stress",
    "Disease_Confounded"
  ],
  "matrix": [
    [
      3,
      2,
      2,
      1
    ],
    [
      1,
      5,
      3,
      0
    ],
    [
      1,
      3,
      7,
      0
    ],
    [
      1,
      0,
      1,
      10
    ]
  ]
}
```

Any non-Healthy prediction among actual Healthy images, including Disease Confounded.

### Support Vector Machine (RBF) — development-only diagnostics

| Metric | Value |
|---|---:|
| accuracy | 0.625 |
| macro_f1 | 0.588474025974026 |
| balanced_accuracy | 0.5934343434343434 |
| macro_precision | 0.5890151515151515 |
| macro_recall | 0.5934343434343434 |
| weighted_f1 | 0.6196428571428572 |
| severe_to_healthy | 1 |
| disease_to_water_stress | 2 |
| healthy_false_alert_rate | 0.75 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.3333333333333333 | 0.25 | 0.2857142857142857 | 8.0 |
| Moderate_Water_Stress | 0.45454545454545453 | 0.5555555555555556 | 0.5 | 9.0 |
| Severe_Water_Stress | 0.8181818181818182 | 0.8181818181818182 | 0.8181818181818182 | 11.0 |
| Disease_Confounded | 0.75 | 0.75 | 0.75 | 12.0 |

Confusion matrix (rows actual, columns predicted; order below):

```json
{
  "labels": [
    "Healthy",
    "Moderate_Water_Stress",
    "Severe_Water_Stress",
    "Disease_Confounded"
  ],
  "matrix": [
    [
      2,
      3,
      1,
      2
    ],
    [
      2,
      5,
      1,
      1
    ],
    [
      1,
      1,
      9,
      0
    ],
    [
      1,
      2,
      0,
      9
    ]
  ]
}
```

Any non-Healthy prediction among actual Healthy images, including Disease Confounded.

### Random Forest — development-only diagnostics

| Metric | Value |
|---|---:|
| accuracy | 0.575 |
| macro_f1 | 0.5016915954415955 |
| balanced_accuracy | 0.5258838383838385 |
| macro_precision | 0.4946886446886447 |
| macro_recall | 0.5258838383838383 |
| weighted_f1 | 0.5410523504273504 |
| severe_to_healthy | 0 |
| disease_to_water_stress | 1 |
| healthy_false_alert_rate | 0.75 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.4 | 0.25 | 0.3076923076923077 | 8.0 |
| Moderate_Water_Stress | 0.14285714285714285 | 0.1111111111111111 | 0.125 | 9.0 |
| Severe_Water_Stress | 0.7692307692307693 | 0.9090909090909091 | 0.8333333333333334 | 11.0 |
| Disease_Confounded | 0.6666666666666666 | 0.8333333333333334 | 0.7407407407407407 | 12.0 |

Confusion matrix (rows actual, columns predicted; order below):

```json
{
  "labels": [
    "Healthy",
    "Moderate_Water_Stress",
    "Severe_Water_Stress",
    "Disease_Confounded"
  ],
  "matrix": [
    [
      2,
      4,
      0,
      2
    ],
    [
      2,
      1,
      3,
      3
    ],
    [
      0,
      1,
      10,
      0
    ],
    [
      1,
      1,
      0,
      10
    ]
  ]
}
```

Any non-Healthy prediction among actual Healthy images, including Disease Confounded.

### K-Nearest Neighbors — development-only diagnostics

| Metric | Value |
|---|---:|
| accuracy | 0.45 |
| macro_f1 | 0.3956043956043956 |
| balanced_accuracy | 0.4160353535353535 |
| macro_precision | 0.39285714285714285 |
| macro_recall | 0.4160353535353535 |
| weighted_f1 | 0.4219780219780219 |
| severe_to_healthy | 1 |
| disease_to_water_stress | 2 |
| healthy_false_alert_rate | 0.75 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.3333333333333333 | 0.25 | 0.2857142857142857 | 8.0 |
| Moderate_Water_Stress | 0.2 | 0.1111111111111111 | 0.14285714285714285 | 9.0 |
| Severe_Water_Stress | 0.4666666666666667 | 0.6363636363636364 | 0.5384615384615384 | 11.0 |
| Disease_Confounded | 0.5714285714285714 | 0.6666666666666666 | 0.6153846153846154 | 12.0 |

Confusion matrix (rows actual, columns predicted; order below):

```json
{
  "labels": [
    "Healthy",
    "Moderate_Water_Stress",
    "Severe_Water_Stress",
    "Disease_Confounded"
  ],
  "matrix": [
    [
      2,
      2,
      2,
      2
    ],
    [
      1,
      1,
      4,
      3
    ],
    [
      1,
      2,
      7,
      1
    ],
    [
      2,
      0,
      2,
      8
    ]
  ]
}
```

Any non-Healthy prediction among actual Healthy images, including Disease Confounded.

### Gradient Boosting — development-only diagnostics

| Metric | Value |
|---|---:|
| accuracy | 0.45 |
| macro_f1 | 0.42122400017136863 |
| balanced_accuracy | 0.4245580808080808 |
| macro_precision | 0.4375 |
| macro_recall | 0.4245580808080808 |
| weighted_f1 | 0.4457232825653878 |
| severe_to_healthy | 0 |
| disease_to_water_stress | 4 |
| healthy_false_alert_rate | 0.875 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.2 | 0.125 | 0.15384615384615385 | 8.0 |
| Moderate_Water_Stress | 0.3333333333333333 | 0.4444444444444444 | 0.38095238095238093 | 9.0 |
| Severe_Water_Stress | 0.75 | 0.5454545454545454 | 0.631578947368421 | 11.0 |
| Disease_Confounded | 0.4666666666666667 | 0.5833333333333334 | 0.5185185185185185 | 12.0 |

Confusion matrix (rows actual, columns predicted; order below):

```json
{
  "labels": [
    "Healthy",
    "Moderate_Water_Stress",
    "Severe_Water_Stress",
    "Disease_Confounded"
  ],
  "matrix": [
    [
      1,
      2,
      1,
      4
    ],
    [
      3,
      4,
      0,
      2
    ],
    [
      0,
      3,
      6,
      2
    ],
    [
      1,
      3,
      1,
      7
    ]
  ]
}
```

Any non-Healthy prediction among actual Healthy images, including Disease Confounded.
