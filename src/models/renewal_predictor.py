from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.features.build_renewal_features import build_renewal_features

try:
    import shap
except ImportError:  # pragma: no cover - optional in some local envs
    shap = None


class ModelNotTrainedError(FileNotFoundError):
    """Raised when renewal prediction is requested before a model artifact exists."""


class RenewalPredictor:
    """Shared prediction service used by the API, dashboard, and monitoring."""

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        config_path: str | Path = "params.yaml",
    ) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        training_config = self.config["training"]
        raw_config = self.config["raw_data_config"]

        artifact = artifact_path or training_config["artifact_path"]
        self.artifact_path = resolve_path(artifact, config_path)
        if not self.artifact_path.exists():
            raise ModelNotTrainedError(
                f"Model artifact not found at {self.artifact_path}."
            )

        bundle = joblib.load(self.artifact_path)
        self.model = bundle["model"]
        self.metadata = bundle.get("metadata", {})
        self.target = self.metadata.get("target", raw_config["target"])
        self.feature_names = self.metadata.get(
            "feature_names",
            self.config["feature_engineering"]["model_features"],
        )
        self.input_features = self.metadata.get(
            "input_features",
            raw_config["input_features"],
        )
        self.positive_label = int(self.metadata.get("positive_label", 1))
        self.prediction_threshold = float(
            self.metadata.get(
                "prediction_threshold",
                training_config.get("prediction_threshold", 0.5),
            )
        )
        raw_label_map = self.metadata.get(
            "label_map",
            {0: "not_renewed", 1: "renewed"},
        )
        self.label_map = {int(key): value for key, value in raw_label_map.items()}
        self.training_feature_medians = {
            key: float(value)
            for key, value in self.metadata.get("training_feature_medians", {}).items()
        }
        self._explainer = None

    def _ensure_model_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if set(self.feature_names).issubset(frame.columns):
            return frame[self.feature_names].astype(float)

        missing_inputs = [column for column in self.input_features if column not in frame.columns]
        if missing_inputs:
            raise ValueError(
                "Missing required subscription inputs: "
                + ", ".join(missing_inputs)
            )

        enriched = build_renewal_features(frame[self.input_features], config=self.config)
        return enriched[self.feature_names].astype(float)

    def _to_frame(self, records: Any) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, dict):
            frame = pd.DataFrame([records])
        else:
            frame = pd.DataFrame(records)
        return self._ensure_model_frame(frame)

    def _positive_class_index(self) -> int:
        classes = [int(value) for value in getattr(self.model, "classes_", [])]
        if self.positive_label in classes:
            return classes.index(self.positive_label)
        if len(classes) > 1:
            return 1
        return 0

    def _format_prediction(self, label: int, probability: float) -> dict[str, float | int | str]:
        return {
            "renewal_prediction": int(label),
            "renewal_probability": float(probability),
            "renewal_label": self.label_map.get(int(label), str(label)),
        }

    def predict_proba(self, records: Any) -> np.ndarray:
        frame = self._to_frame(records)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(frame)
            return probabilities[:, self._positive_class_index()]

        predictions = self.model.predict(frame)
        return np.asarray(
            [1.0 if int(prediction) == self.positive_label else 0.0 for prediction in predictions],
            dtype=float,
        )

    def predict(self, records: Any) -> list[dict[str, float | int | str]]:
        frame = self._to_frame(records)
        probabilities = self.predict_proba(frame)
        labels = (probabilities >= self.prediction_threshold).astype(int)
        return [
            self._format_prediction(label=int(label), probability=float(probability))
            for label, probability in zip(labels, probabilities)
        ]

    def predict_one(self, record: dict[str, Any]) -> dict[str, float | int | str]:
        return self.predict(record)[0]

    def _get_explainer(self):
        if shap is None:
            raise RuntimeError("SHAP is not installed.")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def _compute_shap_values(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, float | None]:
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(frame)
        expected_value = getattr(explainer, "expected_value", None)

        if isinstance(shap_values, list):
            class_index = self._positive_class_index()
            values = np.asarray(shap_values[class_index])
            if isinstance(expected_value, (list, tuple, np.ndarray)):
                expected_value = float(np.asarray(expected_value)[class_index])
            return values, expected_value

        values = np.asarray(shap_values)
        if values.ndim == 3:
            class_index = self._positive_class_index()
            values = values[:, :, class_index]
            if isinstance(expected_value, (list, tuple, np.ndarray)):
                expected_value = float(np.asarray(expected_value)[class_index])
        elif isinstance(expected_value, (list, tuple, np.ndarray)):
            expected_value = float(np.asarray(expected_value).ravel()[0])

        return values, expected_value

    def _fallback_contributions(self, frame: pd.DataFrame) -> list[dict[str, float | str]]:
        importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)))
        median_series = pd.Series(self.training_feature_medians)
        contributions = []
        for index, feature_name in enumerate(self.feature_names):
            baseline = float(median_series.get(feature_name, 0.0))
            feature_value = float(frame.iloc[0][feature_name])
            delta = feature_value - baseline
            contributions.append(
                {
                    "feature": feature_name,
                    "feature_value": feature_value,
                    "shap_value": float(delta * float(importances[index])),
                }
            )
        return sorted(contributions, key=lambda item: abs(item["shap_value"]), reverse=True)

    def explain(
        self,
        record: dict[str, Any] | pd.DataFrame,
        top_k: int = 5,
    ) -> dict[str, Any]:
        frame = self._to_frame(record)
        prediction = self.predict(frame)[0]

        try:
            shap_values, expected_value = self._compute_shap_values(frame)
            contributions = []
            row_values = frame.iloc[0]
            for feature_name, feature_value, shap_value in zip(
                self.feature_names,
                row_values.tolist(),
                shap_values[0].tolist(),
            ):
                contributions.append(
                    {
                        "feature": feature_name,
                        "feature_value": float(feature_value),
                        "shap_value": float(shap_value),
                    }
                )
            ranked = sorted(
                contributions,
                key=lambda item: abs(item["shap_value"]),
                reverse=True,
            )
            backend = "shap"
        except Exception:
            ranked = self._fallback_contributions(frame)
            expected_value = None
            backend = "feature_importance_fallback"

        return {
            "prediction": prediction,
            "base_value": expected_value,
            "explanation_backend": backend,
            "top_factors": ranked[:top_k],
        }
