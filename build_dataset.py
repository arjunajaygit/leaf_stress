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

import hashlib
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
seen_hashes = {}   # md5 hex -> (label, filename) of first occurrence
duplicates_removed = 0

for label in CLASSES:
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found, skipping: {folder}")
        continue

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        path = os.path.join(folder, fname)

        # Content-hash deduplication: skip files with identical bytes
        file_hash = hashlib.md5(open(path, "rb").read()).hexdigest()
        if file_hash in seen_hashes:
            orig_label, orig_fname = seen_hashes[file_hash]
            print(f"  duplicate: {path} (identical to {orig_label}/{orig_fname})")
            duplicates_removed += 1
            continue
        seen_hashes[file_hash] = (label, fname)

        feats = extract_all_features(path)
        if feats is None:
            print(f"  skipped (no leaf detected): {path}")
            continue
        feats["label"] = label
        feats["filename"] = fname
        rows.append(feats)

df = pd.DataFrame(rows)
df.to_csv("leaf_features.csv", index=False)

print(f"\nDuplicates removed (by MD5 content hash): {duplicates_removed}")
print(f"Saved {len(df)} unique rows to leaf_features.csv")
print(df["label"].value_counts())
