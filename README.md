# PhytoStress AI v2.0 🍃
### Abiotic Drought Stress & Foliar Pathology Diagnostic Suite
*A Computer Vision & Machine Learning Framework for Phenotypic Plant Stress Assessment*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.9%2B-orange.svg)](https://scikit-learn.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.10%2B-green.svg)](https://opencv.org)
[![Flask](https://img.shields.io/badge/Flask-v3.1-black.svg)](https://flask.palletsprojects.com)

---

## 📌 Executive Summary & Academic Scope

In precision agronomy, early detection of plant water deficit is essential to prevent permanent wilting points and catastrophic yield loss. However, foliar water stress symptoms (wilting, chlorosis, and marginal necrosis) frequently overlap with biotic fungal and bacterial infections.

**PhytoStress AI v2.0** solves this challenge through a multi-stage, interpretable computer vision and machine learning pipeline that:
1. Segments leaf lamina from uncalibrated field photography.
2. Extracts **15 domain-specific biophysical & morphometric features**.
3. Classifies foliage into 4 distinct phenotypic categories using a **5-model benchmarked ensemble**.
4. Provides **Explainable AI (XAI)** biophysical trait attribution to explain *why* the diagnosis was made.
5. Quantifies stress on a continuous **Crop Water Stress Index (CWSI, 0–100%)** scale with actionable agronomic recovery prescriptions.

---

## 🎯 Target Phenotypic Classes

| Class | Description | Primary Visual & Biophysical Marker |
|---|---|---|
| **Healthy** | Optimal turgor pressure and vegetative vigor | High Green Leaf Index ($\text{GLI} > 0.15$), high solidity, uniform hue |
| **Moderate Water Stress** | Incipient water deficit, early stomatal regulation | Margin curling, drop in compactness, mild chlorosis |
| **Severe Water Stress** | Critical cellular dehydration, flaccidity | Extreme loss of solidity, marginal crisping, tissue desiccation |
| **Disease Confounded** | Biotic infection mimicking drought symptoms | Irregular necrotic spots, high hue dispersion ($\sigma_H$), high contrast ($\sigma_V$) |

---

## 🔬 Mathematical Feature Engineering Pipeline

Rather than deploying unconstrained deep neural networks that memorize small sample sets, PhytoStress AI extracts 15 mathematically verified biophysical features:

### 1. Spectral Vegetation Indices
* **Excess Green Index (ExG):**
  $$\text{ExG} = 2G - R - B$$
  *Isolates green foliar biomass from background noise and specular glares.*
* **Green Leaf Index (GLI):**
  $$\text{GLI} = \frac{2G - R - B}{2G + R + B + 10^{-6}}$$
  *Normalizes chlorophyll reflectance; serves as an optical proxy for foliar nitrogen and hydration.*

### 2. Chromatic Distribution Descriptors
* **HSV Color Distributions:** Mean ($\mu_H, \mu_S, \mu_V$) and standard deviation ($\sigma_H, \sigma_S, \sigma_V$).
  * A spike in $\sigma_H$ (Hue variance) captures pathological lesions and mottled chlorosis, differentiating fungal spots from uniform drought yellowing.

### 3. Morphometric & Curling Descriptors (Turgor Pressure Proxies)
* **Convex Hull Solidity:**
  $$\text{Solidity} = \frac{\text{Contour Area}}{\text{Convex Hull Area}}$$
  *Measures margin involution; wilting leaves collapse inward, creating concavities that sharply lower solidity.*
* **Contour Compactness / Form Factor:**
  $$\text{Compactness} = \frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$$
  *Quantifies outline regularity; leaves undergoing severe dehydration display jagged, rolled borders.*
* **Extent & Aspect Ratio:**
  $$\text{Extent} = \frac{\text{Area}}{W \times H}, \quad \text{Aspect Ratio} = \frac{W}{H}$$

---

## 📊 Academic 5-Fold Stratified Cross-Validation Benchmark

Models evaluated on the same 5-fold stratified cross-validation splits ($N=55$):

| Model Architecture | Paradigm | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (RBF)** | Kernel Maximum Margin | **78.2%** | **78.3%** | **71.7%** | **0.726** | **0.777** |
| **Logistic Regression (L2)** | Regularized Linear | **74.5%** | **74.2%** | **67.6%** | **0.683** | **0.748** |
| **Random Forest (250 Trees)** | Bagged Decision Ensemble | **69.1%** | **55.0%** | **57.5%** | **0.560** | **0.659** |
| **Gradient Boosting (GBM)** | Sequential Gradient Descent | **67.3%** | **57.7%** | **60.9%** | **0.581** | **0.659** |
| **K-Nearest Neighbors (k=5)** | Non-parametric Metric Space | **69.1%** | **57.8%** | **55.3%** | **0.544** | **0.652** |

> **Capstone Viva Defense (Small-Sample Regime):**  
> *"In controlled agronomic drought experiments, acquiring ground-truth calibrated drought samples is resource-intensive ($N=55$). Deep learning architectures (ResNet, MobileNet) have millions of parameters and overfit severely on small sample sizes. By injecting domain knowledge through 15 biophysical features and training regularized Support Vector Machines with RBF kernel, we maximize generalization bounds without requiring millions of training images."*

---

## 🧠 Explainable AI (XAI): Biophysical Trait Attribution

For every scanned leaf, the system calculates the normalized Z-score anomaly against baseline population distributions:

$$Z_j = \frac{x_j - \mu_j}{\sigma_j}$$

Each anomaly is weighted by Random Forest Mean Decrease in Impurity (MDI) to output human-readable, agronomically meaningful explanations:
* *"Solidity: -2.1σ below healthy baseline (Severe leaf edge curling detected)"*
* *"Green Leaf Index: -1.8σ below normal (Substantial photosynthetic degradation)"*
* *"Hue Standard Deviation: +2.4σ above baseline (Irregular foliar lesion spotting detected)"*

---

## 💻 System Architecture & Web Suite

The application features a responsive dashboard designed for desktop and field smartphone use:
* **Diagnostics Scanner:** Instant photo upload, CWSI gauge, and actionable irrigation/mulching instructions.
* **Computer Vision Pipeline Inspector:** Live visualization of (1) Raw Capture, (2) Morphological Cutout, (3) Pathology Tissue Map (Chlorosis vs Necrosis), and (4) GLI Spectral Heatmap.
* **Explainable AI Tab:** Real-time biophysical trait attribution waterfall.
* **Model Benchmark Suite:** Interactive cross-validation metric tables, feature importance bars, and confusion matrices.
* **PDF Diagnostic Dossier:** Client-side printable agronomy report with full recovery checklist.

---

## 🚀 Quickstart & Installation

### 1. Clone & Checkout v2 Branch
```bash
git clone https://github.com/arjunajaygit/leaf_stress.git
cd leaf_stress
git checkout v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Extract Features & Train Models
```bash
# Extract biophysical features from dataset/
python build_dataset.py

# Run 5-fold cross-validation benchmarks and compile model bundle
python train_model.py
```

### 4. Launch the Web Application
```bash
python app.py
```
* **Local browser:** [http://localhost:5001](http://localhost:5001)
* **Mobile device:** Access via `http://<your-computer-ip>:5001` on the same Wi-Fi.

---

## 📁 Repository Structure

```text
leaf_stress/
├── app.py                     # Production Flask web system & multi-model API
├── build_dataset.py           # Feature extraction batch pipeline
├── feature_extraction.py      # Biophysical algorithms, CV generators & CWSI logic
├── train_model.py             # 5-Model cross-validation benchmark & bundle builder
├── requirements.txt           # Python dependencies
├── model_benchmark.json       # Academic cross-validation benchmark metrics
├── leaf_stress_model.pkl      # Serialized model & XAI metadata bundle
├── leaf_features.csv          # 15-dimensional extracted biophysical dataset
├── dataset/                   # Categorized foliar image directories
│   ├── Healthy/
│   ├── Moderate_Water_Stress/
│   ├── Severe_Water_Stress/
│   └── Disease_Confounded/
├── templates/
│   └── index.html             # High-end interactive research dashboard
└── uploads/                   # Runtime inspection cache
```

---

## 🎓 Academic Presentation Tips

When presenting this project to external examiners:
1. **Highlight the Visual Decomposition:** Switch to the **Vision Pipeline** tab to show that the system actually understands leaf contours, chlorosis, and necrosis rather than operating as a black box.
2. **Present the Multi-Model Benchmark:** Show the **5-Fold Cross-Validation** comparison table to prove why SVM with RBF kernel is mathematically superior to heuristic methods on tabular plant data.
3. **Walk through the Explainable AI (XAI) Attributions:** Demonstrate how the Z-score deviations correlate with biological wilting and chlorophyll degradation.