"""
app.py
------
Flask web app. Serves a page where you upload/take a leaf photo on your
phone, runs it through the same feature extraction + trained model,
and shows the predicted stress class.

Run with:  python app.py
Then open  http://<your-computer's-local-IP>:5001  on your phone
(must be on the same Wi-Fi network).
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

from feature_extraction import extract_all_features

app = Flask(__name__)

MODEL_PATH = "leaf_stress_model.pkl"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    feats = extract_all_features(save_path)
    if feats is None:
        return jsonify({"error": "Could not detect a leaf in this photo. Try a clearer, closer shot."}), 400

    feature_cols = list(model.feature_names_in_)
    X = pd.DataFrame([feats])[feature_cols]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    probs = {cls: round(float(p), 3) for cls, p in zip(model.classes_, proba)}

    return jsonify({"prediction": pred, "probabilities": probs})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)

