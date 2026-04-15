from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.features.build_renewal_features import build_renewal_features


def load_data(data_path: str | Path, source_columns: list[str]) -> pd.DataFrame:
    """Load the SaaS subscription fields required by the renewal pipeline."""

    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    return df[source_columns]


def load_raw_data(config_path: str | Path = "params.yaml") -> None:
    """Materialize the feature-engineered renewal dataset into the raw zone."""

    config = load_config(config_path)
    external_data_path = resolve_path(
        config["external_data_config"]["external_data_csv"],
        config_path,
    )
    raw_data_path = resolve_path(config["raw_data_config"]["raw_data_csv"], config_path)
    source_columns = config["raw_data_config"]["source_columns"]

    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_data(external_data_path, source_columns)
    feature_frame = build_renewal_features(df, config=config)
    feature_frame.to_csv(raw_data_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)
