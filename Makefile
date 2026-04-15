.PHONY: clean requirements lint train monitor

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = subscription-renewal-system
PYTHON_INTERPRETER = python3

requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

train: requirements
	$(PYTHON_INTERPRETER) src/data/load_data.py --config params.yaml
	$(PYTHON_INTERPRETER) src/data/split_data.py --config params.yaml
	$(PYTHON_INTERPRETER) src/models/train_renewal_model.py --config params.yaml

monitor: requirements
	$(PYTHON_INTERPRETER) src/monitoring/renewal_drift_report.py --config params.yaml

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 src tests dashboard app.py
