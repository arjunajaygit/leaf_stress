"""
build_dataset.py
-----------------
Walks your labeled image folders, extracts features from each photo,
and saves everything to leaf_features.csv for training.

Expected folder layout (change DATASET_DIR below to point at yours):

    dataset/
        Healthy/
        Moderate_Water_Stress/
        Severe_Water_Stress/
        Disease_Confounded/
"""

import os
import pandas as pd
from feature_extraction import extract_all_features

DATASET_DIR = "dataset"  # <-- point this at your actual folder
CLASSES = [
    "Healthy",
    "Moderate_Water_Stress",
    "Severe_Water_Stress",
    "Disease_Confounded",
]

rows = []
for label in CLASSES:
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found, skipping: {folder}")
        continue

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, fname)
        feats = extract_all_features(path)
        if feats is None:
            print(f"  skipped (no leaf detected): {path}")
            continue
        feats["label"] = label
        feats["filename"] = fname
        rows.append(feats)

df = pd.DataFrame(rows)
df.to_csv("leaf_features.csv", index=False)

print(f"\nSaved {len(df)} rows to leaf_features.csv")
print(df["label"].value_counts())
