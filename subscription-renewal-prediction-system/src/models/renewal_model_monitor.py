"""Wrapper around the renewal drift monitoring job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.renewal_drift_report import generate_renewal_drift_report


def renewal_model_monitoring(config_path: str | Path = "params.yaml") -> dict:
    return generate_renewal_drift_report(config_path=config_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    renewal_model_monitoring(config_path=parsed_args.config)
