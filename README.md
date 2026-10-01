# Leaf Stress Detector 🍃

A computer vision and machine learning application to classify water stress and health in plant/tomato leaves.

## Classes
- **Healthy**
- **Moderate Water Stress**
- **Severe Water Stress**
- **Disease Confounded**

## Features
- Computer vision feature extraction using OpenCV (color spaces HSV, RGB, indices EXG & GLI, contour shape descriptors like solidity, aspect ratio, extent, compactness).
- Random Forest classification using scikit-learn.
- Interactive Flask web app for live leaf photo upload and stress diagnosis with prediction confidence breakdown.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Extract Features**:
   ```bash
   python build_dataset.py
   ```
   Extracts features from images in `dataset/` and outputs `leaf_features.csv`.

2. **Train Model**:
   ```bash
   python train_model.py
   ```
   Evaluates cross-validation performance and saves `leaf_stress_model.pkl`.

3. **Run Web App**:
   ```bash
   python app.py
   ```
   Open `http://localhost:5001` or access from mobile device on the same local Wi-Fi.