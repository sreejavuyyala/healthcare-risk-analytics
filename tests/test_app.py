import os

import pytest

from app import app as flask_app

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "risk_model.pkl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="models/risk_model.pkl not found; run `python -m src.train` first",
)

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0.0, "thal": 6.0,
}


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        yield client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["model_type"] == "RandomForestClassifier"
    assert "training_metrics" in body
    assert set(body["raw_input_features"]) == set(SAMPLE_PATIENT.keys())


def test_predict_returns_risk_score_and_category(client):
    resp = client.post("/predict", json=SAMPLE_PATIENT)
    assert resp.status_code == 200
    body = resp.get_json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_category"] in {"Low", "Medium", "High"}
    assert body["prediction"] in {0, 1}


def test_predict_accepts_missing_optional_ca_and_thal(client):
    payload = {k: v for k, v in SAMPLE_PATIENT.items() if k not in {"ca", "thal"}}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_rejects_missing_required_field(client):
    payload = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "age" in resp.get_json()["error"]


def test_predict_rejects_non_json_body(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_rejects_invalid_field_type(client):
    payload = dict(SAMPLE_PATIENT, age="not-a-number")
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
