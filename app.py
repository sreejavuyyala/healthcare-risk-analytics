"""Flask REST API serving the heart disease risk model.

Endpoints:
  POST /predict     patient attributes (JSON) -> risk score + risk category
  GET  /health       liveness check
  GET  /model-info   model metadata, feature list, training metrics
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.data import RAW_FEATURE_COLUMNS
from src.features import risk_category

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "risk_model.pkl")
METADATA_PATH = os.path.join(ROOT_DIR, "models", "model_metadata.json")

# ca and thal have known missing values in the source dataset; the trained
# pipeline imputes them, so callers may omit or null them.
OPTIONAL_FIELDS = {"ca", "thal"}
REQUIRED_FIELDS = [f for f in RAW_FEATURE_COLUMNS if f not in OPTIONAL_FIELDS]

app = Flask(__name__)

_model = None
_metadata = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metadata():
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _metadata


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/model-info", methods=["GET"])
def model_info():
    metadata = get_metadata()
    return jsonify({
        "model_type": metadata["model_type"],
        "best_params": metadata["best_params"],
        "raw_input_features": metadata["raw_input_features"],
        "target_definition": metadata["target_definition"],
        "training_metrics": {
            "baseline": metadata["baseline_metrics"],
            "improved": metadata["improved_metrics"],
            "accuracy_lift_pct_over_baseline": metadata["accuracy_lift_pct_over_baseline"],
        },
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] is None]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    row = {f: payload.get(f) for f in RAW_FEATURE_COLUMNS}

    try:
        X = pd.DataFrame([row], columns=RAW_FEATURE_COLUMNS).astype(float)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid field value: {exc}"}), 400

    model = get_model()
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(model.predict(X)[0])

    return jsonify({
        "risk_score": round(probability, 4),
        "risk_category": risk_category(probability),
        "prediction": prediction,
        "prediction_label": "heart disease present" if prediction == 1 else "heart disease absent",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
