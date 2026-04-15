from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PLAN = "growth"
BASE_NUMERIC_FEATURES = [
    "monthly_usage_hours",
    "login_frequency",
    "last_login_days",
    "support_tickets",
    "payment_failures",
]


def _resolve_feature_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    return config.get("feature_engineering", config)


def normalize_subscription_plan(series: pd.Series) -> pd.Series:
    normalized = (
        series.fillna(DEFAULT_PLAN)
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", DEFAULT_PLAN)
    )
    allowed = {"starter", "growth", "business", "enterprise"}
    return normalized.where(normalized.isin(allowed), DEFAULT_PLAN)


def build_renewal_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    feature_config = _resolve_feature_config(config)
    plan_scores = feature_config.get(
        "plan_scores",
        {"starter": 0.7, "growth": 1.0, "business": 1.3, "enterprise": 1.6},
    )
    inactivity_weight = float(feature_config.get("inactivity_risk_weight", 0.12))
    payment_weight = float(feature_config.get("payment_failure_risk_weight", 1.75))
    support_weight = float(feature_config.get("support_ticket_risk_weight", 0.55))
    engagement_weight = float(feature_config.get("engagement_relief_weight", 0.85))

    frame = df.copy()
    for column in BASE_NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    frame["subscription_plan"] = normalize_subscription_plan(frame["subscription_plan"])
    frame["plan_value_index"] = frame["subscription_plan"].map(plan_scores).astype(float)
    frame["engagement_score"] = frame["monthly_usage_hours"] / (frame["last_login_days"] + 1.0)
    frame["activity_ratio"] = frame["login_frequency"] / (frame["monthly_usage_hours"] + 1.0)
    frame["support_pressure"] = frame["support_tickets"] / (frame["login_frequency"] + 1.0)
    frame["payment_reliability"] = 1.0 / (frame["payment_failures"] + 1.0)
    frame["usage_momentum"] = frame["monthly_usage_hours"] * np.log1p(frame["login_frequency"])

    raw_risk = (
        frame["payment_failures"] * payment_weight
        + frame["last_login_days"] * inactivity_weight
        + frame["support_tickets"] * support_weight
        - frame["engagement_score"] * engagement_weight
        - frame["plan_value_index"] * 0.35
    )
    frame["risk_score"] = raw_risk.clip(lower=0.0).round(4)

    model_features = feature_config.get(
        "model_features",
        BASE_NUMERIC_FEATURES
        + [
            "plan_value_index",
            "engagement_score",
            "activity_ratio",
            "support_pressure",
            "payment_reliability",
            "usage_momentum",
            "risk_score",
        ],
    )
    passthrough = [
        column
        for column in frame.columns
        if column not in model_features and column != "subscription_plan"
    ]
    return frame[passthrough + model_features]
