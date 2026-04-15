"""Sync the trained renewal artifact to the serving path."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path


def sync_renewal_serving_model(config_path: str | Path = "params.yaml") -> str:
    config = load_config(config_path)
    source = resolve_path(config["training"]["artifact_path"], config_path)
    destination = resolve_path(config["model_dir"], config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    sync_renewal_serving_model(config_path=parsed_args.config)
