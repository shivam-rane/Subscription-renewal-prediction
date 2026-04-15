from __future__ import annotations

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import GradientBoostingClassifier

from src.api import app as api_module
from src.features.build_renewal_features import build_renewal_features
from src.models.renewal_predictor import RenewalPredictor

BASE_FEATURES = [
    "monthly_usage_hours",
    "login_frequency",
    "last_login_days",
    "support_tickets",
    "payment_failures",
    "subscription_plan",
]

MODEL_FEATURES = [
    "monthly_usage_hours",
    "login_frequency",
    "last_login_days",
    "support_tickets",
    "payment_failures",
    "plan_value_index",
    "engagement_score",
    "activity_ratio",
    "support_pressure",
    "payment_reliability",
    "usage_momentum",
    "risk_score",
]


@pytest.fixture()
def sample_account() -> dict[str, float | str]:
    return {
        "monthly_usage_hours": 42.0,
        "login_frequency": 18.0,
        "last_login_days": 2.0,
        "support_tickets": 1.0,
        "payment_failures": 0.0,
        "subscription_plan": "business",
    }


@pytest.fixture()
def predictor(tmp_path) -> RenewalPredictor:
    training_frame = pd.DataFrame(
        [
            [18.0, 7.0, 11.0, 3.0, 2.0, "starter", 0],
            [24.0, 9.0, 8.0, 2.0, 1.0, "starter", 0],
            [36.0, 14.0, 5.0, 1.0, 1.0, "growth", 1],
            [48.0, 19.0, 2.0, 1.0, 0.0, "business", 1],
            [61.0, 27.0, 1.0, 0.0, 0.0, "enterprise", 1],
            [28.0, 10.0, 7.0, 2.0, 1.0, "growth", 0],
        ],
        columns=BASE_FEATURES + ["renewed"],
    )
    model_frame = build_renewal_features(training_frame[BASE_FEATURES])
    model = GradientBoostingClassifier(random_state=7, n_estimators=20)
    model.fit(model_frame[MODEL_FEATURES], training_frame["renewed"])

    artifact_path = tmp_path / "model.joblib"
    bundle = {
        "model": model,
        "metadata": {
            "feature_names": MODEL_FEATURES,
            "input_features": BASE_FEATURES,
            "target": "renewed",
            "positive_label": 1,
            "prediction_threshold": 0.5,
            "label_map": {0: "not_renewed", 1: "renewed"},
            "training_feature_medians": model_frame[MODEL_FEATURES].median().to_dict(),
        },
    }
    joblib.dump(bundle, artifact_path)
    return RenewalPredictor(artifact_path=artifact_path)


@pytest.fixture()
def client(predictor):
    api_module._predictor = predictor
    api_module._request_count = 0
    api_module._error_count = 0
    return TestClient(api_module.app)


def test_predictor_returns_probability(predictor, sample_account):
    result = predictor.predict_one(sample_account)
    assert result["renewal_prediction"] in {0, 1}
    assert result["renewal_label"] in {"renewed", "not_renewed"}
    assert 0.0 <= result["renewal_probability"] <= 1.0


def test_predict_endpoint_returns_prediction(client, sample_account):
    response = client.post("/predict-renewal", json=sample_account)

    assert response.status_code == 200
    payload = response.json()
    assert payload["renewal_prediction"] in {0, 1}
    assert 0.0 <= payload["renewal_probability"] <= 1.0


def test_batch_predict_endpoint_returns_multiple_predictions(client, sample_account):
    response = client.post(
        "/predict-renewal/batch",
        json={"accounts": [sample_account, sample_account]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 2


def test_explain_endpoint_returns_top_factors(client, sample_account):
    response = client.post("/explain-renewal", json=sample_account)

    assert response.status_code == 200
    payload = response.json()
    assert "top_factors" in payload
    assert len(payload["top_factors"]) > 0


def test_predict_validation_rejects_bad_payload(client, sample_account):
    invalid = dict(sample_account)
    invalid["unexpected"] = 1

    response = client.post("/predict-renewal", json=invalid)

    assert response.status_code == 422


def test_metrics_endpoint_exposes_custom_metrics(client, sample_account):
    client.post("/predict-renewal", json=sample_account)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "renewal_prediction_requests_total" in response.text
    assert "renewal_model_latency_seconds" in response.text
