"""
train_model.py
---------------
Trains a Random Forest on leaf_features.csv.

With only ~40 images total, a single train/test split would be too noisy
to trust, so this uses stratified k-fold cross-validation to evaluate, then
trains one final model on all the data for the web app to use.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("leaf_features.csv")
feature_cols = [c for c in df.columns if c not in ("label", "filename")]
X = df[feature_cols]
y = df["label"]

print("Class counts:")
print(y.value_counts(), "\n")

# Can't use more folds than the smallest class has samples
n_splits = min(5, y.value_counts().min())
if n_splits < 2:
    raise ValueError(
        "At least one class has fewer than 2 images — add more samples "
        "or merge that class before training."
    )

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)

y_pred = cross_val_predict(clf, X, y, cv=cv)

print(f"Cross-validated results ({n_splits}-fold):\n")
print(classification_report(y, y_pred))
print("Confusion matrix (rows=true, cols=predicted):")
labels_sorted = sorted(y.unique())
print(pd.DataFrame(
    confusion_matrix(y, y_pred, labels=labels_sorted),
    index=labels_sorted, columns=labels_sorted,
))

# Train the final model on ALL available data for deployment
clf.fit(X, y)
joblib.dump(clf, "leaf_stress_model.pkl")
print("\nFinal model (trained on all data) saved to leaf_stress_model.pkl")

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:")
print(importances)
