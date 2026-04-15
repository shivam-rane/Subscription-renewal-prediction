from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.models.train_renewal_model import train_and_evaluate


def test_train_and_evaluate_writes_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    external_data = pd.read_csv(
        repo_root / "data" / "external" / "subscription_accounts.csv"
    ).head(240)
    external_csv = tmp_path / "external.csv"
    external_data.to_csv(external_csv, index=False)

    config = {
        "external_data_config": {"external_data_csv": str(external_csv)},
        "raw_data_config": {
            "raw_data_csv": str(tmp_path / "raw" / "train.csv"),
            "source_columns": [
                "monthly_usage_hours",
                "login_frequency",
                "last_login_days",
                "support_tickets",
                "payment_failures",
                "subscription_plan",
                "renewed",
            ],
            "input_features": [
                "monthly_usage_hours",
                "login_frequency",
                "last_login_days",
                "support_tickets",
                "payment_failures",
                "subscription_plan",
            ],
            "train_test_split_ratio": 0.2,
            "target": "renewed",
            "positive_class": 1,
            "random_state": 111,
        },
        "feature_engineering": {
            "plan_scores": {
                "starter": 0.7,
                "growth": 1.0,
                "business": 1.3,
                "enterprise": 1.6,
            },
            "inactivity_risk_weight": 0.12,
            "payment_failure_risk_weight": 1.75,
            "support_ticket_risk_weight": 0.55,
            "engagement_relief_weight": 0.85,
            "model_features": [
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
            ],
        },
        "processed_data_config": {
            "train_data_csv": str(tmp_path / "processed" / "train.csv"),
            "test_data_csv": str(tmp_path / "processed" / "test.csv"),
        },
        "training": {
            "artifact_path": str(tmp_path / "models" / "subscription_renewal_model.joblib"),
            "metrics_path": str(tmp_path / "reports" / "metrics.json"),
            "best_params_path": str(tmp_path / "reports" / "best_params.json"),
            "n_trials": 2,
            "cv_folds": 3,
            "scoring": "roc_auc",
            "prediction_threshold": 0.5,
        },
        "gradient_boosting": {
            "n_estimators": [20, 30],
            "learning_rate": [0.05, 0.15],
            "max_depth": [2, 3],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2],
            "subsample": [0.8, 1.0],
            "max_features": ["sqrt"],
        },
        "model_dir": str(tmp_path / "webapp" / "model.joblib"),
        "drift_monitoring": {
            "reference_data_csv": str(tmp_path / "processed" / "train.csv"),
            "current_data_csv": str(tmp_path / "processed" / "test.csv"),
            "report_html": str(tmp_path / "reports" / "renewal_drift_report.html"),
            "summary_json": str(tmp_path / "reports" / "renewal_drift_report.json"),
        },
    }

    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_and_evaluate(config_path=config_path, n_trials=2)

    assert Path(result["artifact_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["best_params_path"]).exists()
    assert "roc_auc" in result["metrics"]
