from __future__ import annotations

import os
import threading
import time
from typing import Annotated

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from src.models.renewal_predictor import ModelNotTrainedError, RenewalPredictor

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover - fallback for lean local environments
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    _METRIC_NAMES: list[str] = []

    class _DummyMetric:
        def __init__(self, name: str, *_args, **_kwargs):
            self.name = name
            _METRIC_NAMES.append(name)

        def inc(self):
            return None

        def observe(self, _value):
            return None

        def set(self, _value):
            return None

    Counter = Gauge = Histogram = _DummyMetric

    def generate_latest():
        return ("\n".join(_METRIC_NAMES) + "\n").encode("utf-8")

APP_CONFIG_PATH_ENV = "APP_CONFIG_PATH"
MODEL_ARTIFACT_PATH_ENV = "MODEL_ARTIFACT_PATH"

prediction_requests_total = Counter(
    "renewal_prediction_requests_total",
    "Total number of subscription renewal prediction requests served by the API.",
)
prediction_errors_total = Counter(
    "renewal_prediction_errors_total",
    "Total number of failed subscription renewal prediction requests.",
)
model_latency_seconds = Histogram(
    "renewal_model_latency_seconds",
    "Subscription renewal prediction latency in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
model_error_rate = Gauge(
    "renewal_model_error_rate",
    "Fraction of renewal prediction requests that resulted in an error.",
)

_metrics_lock = threading.Lock()
_request_count = 0
_error_count = 0
_predictor: RenewalPredictor | None = None

app = FastAPI(
    title="Subscription Renewal Serving API",
    version="1.0.0",
    description=(
        "FastAPI serving layer for SaaS subscription lifecycle scoring, "
        "batch renewal inference, explanations, and Prometheus metrics."
    ),
)


class SubscriptionAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_usage_hours: Annotated[float, Field(ge=0)]
    login_frequency: Annotated[float, Field(ge=0)]
    last_login_days: Annotated[float, Field(ge=0)]
    support_tickets: Annotated[float, Field(ge=0)]
    payment_failures: Annotated[float, Field(ge=0)]
    subscription_plan: Annotated[str, Field(min_length=3, max_length=32)]


class BatchRenewalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accounts: list[SubscriptionAccount] = Field(..., min_length=1, max_length=1000)


def get_predictor() -> RenewalPredictor:
    global _predictor
    if _predictor is None:
        config_path = os.getenv(APP_CONFIG_PATH_ENV, "params.yaml")
        artifact_path = os.getenv(MODEL_ARTIFACT_PATH_ENV)
        _predictor = RenewalPredictor(
            artifact_path=artifact_path,
            config_path=config_path,
        )
    return _predictor


def _record_metrics(start_time: float, failed: bool) -> None:
    global _request_count, _error_count
    prediction_requests_total.inc()
    if failed:
        prediction_errors_total.inc()
    model_latency_seconds.observe(time.perf_counter() - start_time)

    with _metrics_lock:
        _request_count += 1
        if failed:
            _error_count += 1
        model_error_rate.set(_error_count / _request_count)


def _account_to_dict(account: SubscriptionAccount) -> dict[str, float | str]:
    return account.model_dump()


@app.exception_handler(ModelNotTrainedError)
async def model_not_found_handler(_, exc: ModelNotTrainedError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/")
def root():
    predictor = get_predictor()
    return {
        "service": "subscription-renewal-api",
        "status": "ok",
        "input_features": predictor.input_features,
        "model_features": predictor.feature_names,
        "endpoints": [
            "/predict-renewal",
            "/predict-renewal/batch",
            "/explain-renewal",
            "/metrics",
        ],
    }


@app.get("/health")
def health():
    predictor = get_predictor()
    return {"status": "healthy", "model_path": str(predictor.artifact_path)}


@app.post("/predict-renewal")
def predict_renewal(account: SubscriptionAccount):
    started_at = time.perf_counter()
    failed = False
    try:
        return get_predictor().predict_one(_account_to_dict(account))
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed)


@app.post("/predict-renewal/batch")
def predict_renewal_batch(payload: BatchRenewalRequest):
    started_at = time.perf_counter()
    failed = False
    try:
        accounts = [_account_to_dict(account) for account in payload.accounts]
        return {"predictions": get_predictor().predict(accounts)}
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed)


@app.post("/explain-renewal")
def explain_renewal(account: SubscriptionAccount):
    started_at = time.perf_counter()
    failed = False
    try:
        return get_predictor().explain(_account_to_dict(account))
    except Exception:
        failed = True
        raise
    finally:
        _record_metrics(started_at, failed)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
