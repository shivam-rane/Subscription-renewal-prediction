# Subscription Renewal Prediction System

Production-grade MLOps project for predicting whether a SaaS user will renew at the end of the current subscription cycle. The system models subscription lifecycle health using product engagement, account activity, billing reliability, and support friction signals, then serves renewal forecasts through an API, dashboard, and monitoring stack.

## Problem Statement

Digital product teams care less about generic retention labels and more about renewal intent at the account level. This project predicts `renewed` where:

- `1` means the subscriber is expected to renew
- `0` means the subscriber is at risk of non-renewal

The project is framed around a subscription business and uses SaaS-style behavioral signals such as:

- `monthly_usage_hours`
- `login_frequency`
- `last_login_days`
- `support_tickets`
- `payment_failures`
- `subscription_plan`

Derived renewal features include:

- `engagement_score = monthly_usage_hours / (last_login_days + 1)`
- `activity_ratio = login_frequency / (monthly_usage_hours + 1)`
- `risk_score` built from inactivity, payment failures, and support burden

## Architecture

The repository follows an end-to-end MLOps layout:

1. `data/external/subscription_accounts.csv`
   Synthetic SaaS subscription dataset used as the source asset.
2. `src/data/`
   Schema validation, raw data preparation, and train/test splitting.
3. `src/features/build_renewal_features.py`
   Central feature engineering layer shared by training and inference.
4. `src/models/train_renewal_model.py`
   Renewal model training with cross-validation and artifact packaging.
5. `src/models/renewal_predictor.py`
   Shared inference service for API, dashboard, and monitoring jobs.
6. `src/api/app.py`
   FastAPI scoring layer with batch inference, explanations, and metrics.
7. `src/monitoring/renewal_drift_report.py`
   Data and prediction drift reporting.
8. `dashboard/streamlit_app.py`
   Interactive workbench for single-account and batch forecasting.
9. `monitoring/`
   Prometheus and Grafana assets for operational visibility.

## Model Pipeline

### 1. Data Validation

`src/data/validate_data.py` verifies the subscription dataset contains the expected columns and that the `renewed` target has both classes.

### 2. Data Preparation

`src/data/load_data.py` loads the external SaaS dataset and applies renewal feature engineering before the raw dataset is materialized.

### 3. Feature Engineering

`src/features/build_renewal_features.py` creates the model-ready feature space:

- `plan_value_index`
- `engagement_score`
- `activity_ratio`
- `support_pressure`
- `payment_reliability`
- `usage_momentum`
- `risk_score`

### 4. Train/Test Split

`src/data/split_data.py` creates reproducible train and test partitions stratified on `renewed`.

### 5. Training And Evaluation

`src/models/train_renewal_model.py` trains a `GradientBoostingClassifier`, evaluates classification quality, and stores:

- `models/subscription_renewal_model.joblib`
- `reports/renewal_model_metrics.json`
- `reports/renewal_best_params.json`

### 6. Serving

`src/models/renewal_model_registry.py` copies the production-ready artifact into the serving directory used by the app layer.

### 7. Monitoring

`src/monitoring/renewal_drift_report.py` generates a renewal drift report and a machine-readable summary for monitoring pipelines.

## Tech Stack

- Python 3.11
- scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- Optuna with local fallback when unavailable
- SHAP with feature-importance fallback when unavailable
- Evidently
- Prometheus
- Grafana
- Docker
- DVC
- GitHub Actions
- pytest
- flake8

## Project Structure

```text
.
|-- app.py
|-- params.yaml
|-- dvc.yaml
|-- Dockerfile
|-- dashboard/
|   `-- streamlit_app.py
|-- data/
|   `-- external/subscription_accounts.csv
|-- models/
|   `-- subscription_renewal_model.joblib
|-- monitoring/
|   |-- prometheus.yml
|   `-- grafana/
|-- reports/
|   |-- renewal_model_metrics.json
|   |-- renewal_best_params.json
|   `-- renewal_drift_report.html
|-- src/
|   |-- api/app.py
|   |-- config.py
|   |-- data/
|   |   |-- load_data.py
|   |   |-- split_data.py
|   |   `-- validate_data.py
|   |-- features/
|   |   `-- build_renewal_features.py
|   |-- models/
|   |   |-- renewal_predictor.py
|   |   |-- renewal_model_monitor.py
|   |   |-- renewal_model_registry.py
|   |   `-- train_renewal_model.py
|   `-- monitoring/
|       `-- renewal_drift_report.py
`-- tests/
    |-- test_renewal_api.py
    `-- test_renewal_training.py
```

## How To Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate And Prepare Data

```bash
python src/data/validate_data.py --config params.yaml
python src/data/load_data.py --config params.yaml
python src/data/split_data.py --config params.yaml
```

### 3. Train The Renewal Model

```bash
python src/models/train_renewal_model.py --config params.yaml --n-trials 10
```

### 4. Copy The Serving Artifact

```bash
python src/models/renewal_model_registry.py --config params.yaml
```

### 5. Start The API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Launch The Streamlit Workbench

```bash
streamlit run dashboard/streamlit_app.py
```

### 7. Generate A Drift Report

```bash
python src/monitoring/renewal_drift_report.py --config params.yaml
```

## API Endpoints

- `GET /`
  Service metadata and active feature contract.
- `GET /health`
  Runtime health check and model path.
- `POST /predict-renewal`
  Single-account renewal prediction.
- `POST /predict-renewal/batch`
  Batch subscription renewal prediction.
- `POST /explain-renewal`
  Top feature drivers for a single account.
- `GET /metrics`
  Prometheus metrics for the serving layer.

### Example Request

```json
{
  "monthly_usage_hours": 42.0,
  "login_frequency": 18.0,
  "last_login_days": 2.0,
  "support_tickets": 1.0,
  "payment_failures": 0.0,
  "subscription_plan": "business"
}
```

### Example Response

```json
{
  "renewal_prediction": 1,
  "renewal_probability": 0.91,
  "renewal_label": "renewed"
}
```

## DVC Stages

- `renewal_dataset_creation`
- `renewal_data_split`
- `renewal_model_train`
- `renewal_model_publish`

## Monitoring And Operations

The project includes:

- Prometheus scraping for API metrics
- Grafana dashboard assets for request rate and error-rate visibility
- drift summaries in `reports/renewal_drift_report.json`
- CI/CD workflow for validation, testing, training, monitoring, and image builds

## Verification

The current refactor has been validated locally with:

- `python src/data/validate_data.py --config params.yaml`
- `python src/data/load_data.py --config params.yaml`
- `python src/data/split_data.py --config params.yaml`
- `python src/models/train_renewal_model.py --config params.yaml --n-trials 2`
- `python src/models/renewal_model_registry.py --config params.yaml`
- `python src/monitoring/renewal_drift_report.py --config params.yaml`
- `python -m pytest -q tests`
