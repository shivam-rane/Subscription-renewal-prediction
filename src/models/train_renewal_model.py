from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.data.load_data import load_raw_data
from src.data.split_data import split_and_saved_data

try:
    import optuna
except ImportError:  # pragma: no cover - optional in lightweight local envs
    optuna = None


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

    if not raw_data_path.exists():
        load_raw_data(config_path=config_path)
    if not train_data_path.exists() or not test_data_path.exists():
        split_and_saved_data(config_path=config_path)


def get_search_space(trial, gb_config: dict) -> dict:
    max_features_choices = [
        None if value == "null" else value for value in gb_config["max_features"]
    ]
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            gb_config["n_estimators"][0],
            gb_config["n_estimators"][1],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            gb_config["learning_rate"][0],
            gb_config["learning_rate"][1],
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            gb_config["max_depth"][0],
            gb_config["max_depth"][1],
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            gb_config["min_samples_split"][0],
            gb_config["min_samples_split"][1],
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            gb_config["min_samples_leaf"][0],
            gb_config["min_samples_leaf"][1],
        ),
        "subsample": trial.suggest_float(
            "subsample",
            gb_config["subsample"][0],
            gb_config["subsample"][1],
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            max_features_choices,
        ),
    }


def get_default_params(gb_config: dict) -> dict:
    max_features = gb_config["max_features"][0]
    if max_features == "null":
        max_features = None
    return {
        "n_estimators": gb_config["n_estimators"][0],
        "learning_rate": gb_config["learning_rate"][0],
        "max_depth": gb_config["max_depth"][0],
        "min_samples_split": gb_config["min_samples_split"][0],
        "min_samples_leaf": gb_config["min_samples_leaf"][0],
        "subsample": gb_config["subsample"][0],
        "max_features": max_features,
    }


def evaluate_model(
    model: GradientBoostingClassifier,
    test_x: pd.DataFrame,
    test_y: pd.Series,
) -> dict:
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
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
    requested_trials = n_trials or config["training"]["n_trials"]
    scoring = config["training"].get("scoring", "roc_auc")

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    train_x, train_y = get_feat_and_target(train, target)
    test_x, test_y = get_feat_and_target(test, target)

    gb_config = config["gradient_boosting"]
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    if optuna is not None and requested_trials > 1:
        def objective(trial) -> float:
            search_params = get_search_space(trial, gb_config)
            model = GradientBoostingClassifier(
                random_state=random_state,
                **search_params,
            )
            scores = cross_val_score(
                model,
                train_x,
                train_y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=requested_trials)
        best_params = study.best_params
        best_cv_score = float(study.best_value)
    else:
        best_params = get_default_params(gb_config)
        probe_model = GradientBoostingClassifier(
            random_state=random_state,
            **best_params,
        )
        probe_scores = cross_val_score(
            probe_model,
            train_x,
            train_y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        best_cv_score = float(probe_scores.mean())

    model = GradientBoostingClassifier(
        random_state=random_state,
        **best_params,
    )
    model.fit(train_x, train_y)
    metrics = evaluate_model(model, test_x, test_y)
    metrics["best_cv_score"] = best_cv_score
    metrics["n_trials"] = requested_trials

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
            "prediction_threshold": float(config["training"].get("prediction_threshold", 0.5)),
            "label_map": {0: "not_renewed", 1: "renewed"},
            "training_feature_medians": train_x.median().to_dict(),
        },
        "best_params": best_params,
        "metrics": metrics,
    }
    joblib.dump(bundle, artifact_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    return {
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "best_params_path": str(best_params_path),
        "best_params": best_params,
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
