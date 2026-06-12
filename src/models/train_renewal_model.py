from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.load_data import load_raw_data
from src.data.split_data import split_and_saved_data

def get_feat_and_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target])
    labels = df[target].astype(int)
    return features, labels


def ensure_training_data(config_path: str | Path = "params.yaml") -> None:
    config = load_config(config_path)
    raw_data_path = resolve_path(config["raw_data_config"]["raw_data_csv"], config_path)
    train_data_path = resolve_path(
        config["processed_data_config"]["train_data_csv"],
        config_path,
    )
    test_data_path = resolve_path(
        config["processed_data_config"]["test_data_csv"],
        config_path,
    )
    required_inputs = set(config["raw_data_config"]["input_features"])

    refresh_raw = not raw_data_path.exists()
    if not refresh_raw:
        raw_columns = set(pd.read_csv(raw_data_path, nrows=1).columns)
        refresh_raw = not required_inputs.issubset(raw_columns)
    if refresh_raw:
        load_raw_data(config_path=config_path)

    refresh_processed = not train_data_path.exists() or not test_data_path.exists()
    if not refresh_processed:
        train_columns = set(pd.read_csv(train_data_path, nrows=1).columns)
        test_columns = set(pd.read_csv(test_data_path, nrows=1).columns)
        refresh_processed = not required_inputs.issubset(train_columns) or not required_inputs.issubset(test_columns)
    if refresh_processed:
        split_and_saved_data(config_path=config_path)


def get_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or pd.api.types.is_string_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in features.columns if column not in categorical]
    return numeric, categorical


def build_training_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(C=3.0, max_iter=4000)),
        ]
    )


def select_prediction_threshold(
    model: Pipeline,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    random_state: int,
) -> float:
    fit_x, valid_x, fit_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=0.2,
        random_state=random_state,
        stratify=train_y,
    )
    model.fit(fit_x, fit_y)
    probabilities = model.predict_proba(valid_x)[:, 1]

    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold_int in range(35, 76):
        threshold = threshold_int / 100.0
        predictions = (probabilities >= threshold).astype(int)
        score = accuracy_score(valid_y, predictions)
        if score > best_accuracy:
            best_accuracy = score
            best_threshold = threshold
    return float(best_threshold)


def evaluate_model(
    model: Pipeline,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    prediction_threshold: float,
) -> dict:
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= prediction_threshold).astype(int)
    return {
        "accuracy": accuracy_score(test_y, predictions),
        "precision": precision_score(test_y, predictions, zero_division=0),
        "recall": recall_score(test_y, predictions, zero_division=0),
        "f1": f1_score(test_y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(test_y, probabilities),
        "average_precision": average_precision_score(test_y, probabilities),
        "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
        "classification_report": classification_report(
            test_y,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_evaluate(
    config_path: str | Path = "params.yaml",
    n_trials: int | None = None,
) -> dict:
    ensure_training_data(config_path)
    config = load_config(config_path)
    train_data_path = resolve_path(
        config["processed_data_config"]["train_data_csv"],
        config_path,
    )
    test_data_path = resolve_path(
        config["processed_data_config"]["test_data_csv"],
        config_path,
    )
    artifact_path = resolve_path(config["training"]["artifact_path"], config_path)
    metrics_path = resolve_path(config["training"]["metrics_path"], config_path)
    best_params_path = resolve_path(config["training"]["best_params_path"], config_path)

    target = config["raw_data_config"]["target"]
    positive_label = int(config["raw_data_config"].get("positive_class", 1))
    random_state = config["raw_data_config"]["random_state"]
    cv_folds = config["training"]["cv_folds"]
    scoring = config["training"].get("scoring", "roc_auc")

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    train_x, train_y = get_feat_and_target(train, target)
    test_x, test_y = get_feat_and_target(test, target)

    numeric_features, categorical_features = get_feature_types(train_x)
    pipeline = build_training_pipeline(numeric_features, categorical_features)
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    probe_scores = cross_val_score(
        pipeline,
        train_x,
        train_y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    best_cv_score = float(probe_scores.mean())
    prediction_threshold = select_prediction_threshold(
        model=pipeline,
        train_x=train_x,
        train_y=train_y,
        random_state=random_state,
    )

    model = build_training_pipeline(numeric_features, categorical_features)
    model.fit(train_x, train_y)
    metrics = evaluate_model(model, test_x, test_y, prediction_threshold=prediction_threshold)
    metrics["best_cv_score"] = best_cv_score
    metrics["prediction_threshold"] = prediction_threshold
    metrics["n_trials"] = 1

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "metadata": {
            "feature_names": train_x.columns.tolist(),
            "input_features": config["raw_data_config"]["input_features"],
            "target": target,
            "positive_label": positive_label,
            "prediction_threshold": prediction_threshold,
            "label_map": {0: "not_renewed", 1: "renewed"},
            "training_feature_medians": train_x.select_dtypes(include="number").median().to_dict(),
        },
        "best_params": {
            "model_type": "logistic_regression",
            "polynomial_degree": 2,
            "regularization_c": 3.0,
            "threshold_selection": "validation_accuracy",
            "selected_threshold": prediction_threshold,
        },
        "metrics": metrics,
    }
    best_params = bundle["best_params"]
    joblib.dump(bundle, artifact_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    return {
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "best_params_path": str(best_params_path),
        "best_params": bundle["best_params"],
        "metrics": metrics,
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    args.add_argument("--n-trials", type=int, default=None)
    parsed_args = args.parse_args()
    train_and_evaluate(
        config_path=parsed_args.config,
        n_trials=parsed_args.n_trials,
    )
