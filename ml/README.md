# Machine Learning Engineering Core

> **Scope** This directory houses the core Machine Learning pipelines for the sales forecasting system. It leverages LightGBM for regression, Optuna for hyperparameter tuning, and MLflow for experiment tracking. These scripts are designed to be run locally via the MLflow CLI.

---

## Overview

The ml module standardizes the model building lifecycle using an MLproject definition. The flow covers extracting processed features from the Iceberg Gold layer via Trino. Operations include splitting data based on time cutoffs, discovering hyperparameters via Optuna, and producing a finalized LightGBM model artifact with pkl extension.

Experiment tracking occurs in MLflow with nested runs for Optuna trials to maintain traceable logs.

---

## Directory Layout

```
ml/
├── MLproject                   ← MLflow project definition (entry points)
├── pyproject.toml              ← Python dependencies (managed by uv)
├── processing/
│   └── validator.py            ← Feature column definitions and target mapping
├── scripts/
│   ├── prepare_data.py         ← Trino extraction and Train/Valid/Test splitting
│   ├── train.py                ← Final model training script
│   └── tune.py                 ← Optuna hyperparameter study script
├── tuning/
│   └── objective.py            ← Optuna trial evaluation logic (nested MLflow runs)
└── utils/
    └── mlflow_utils.py         ← Centralised MLflow tracking URI & S3 setup
```

---

## Responsibilities

1. **prepare_data** Connects to the Trino gateway to join Gold layer tables including fact_sales_item_daily and fact_store_weather_daily. This provides over 74 features for the training session and applies Kaggle test ID masking.
2. **tune** Reads dataset files and executes an Optuna study to optimize model parameters. Each trial is logged as a child run under a parent MLflow study.
3. **train** Fits a final LightGBM Regressor based on best parameters. The resulting model is logged to the MLflow Model Registry.

---

## Configuration

This module relies on environment variables loaded from the root `.env`.

### MLflow Tracking & Artifacts
| Variable | Usage | Default Fallback |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Points to the MLflow tracking server | `http://127.0.0.1:5000` |
| `MLFLOW_S3_ENDPOINT_URL` | Endpoint for MinIO (where MLflow stores artifacts) | None |
| `AWS_ACCESS_KEY_ID` | MinIO access key for artifact upload | `minioadmin` |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key for artifact upload | `minioadmin` |

### Lakehouse Connection Trino
| Variable | Usage | Default |
|---|---|---|
| `TRINO_USER` | Trino User | `admin` |
| `TRINO_HOST` | Trino Host | `localhost` |
| `TRINO_PORT` | Trino Port | `8085` |
| `TRINO_CATALOG` | Iceberg Catalog | `iceberg` |
| `TRINO_SCHEMA` | Analytics Schema | `analytics` |

---

## How to Run (Local)

The Python environment is strictly managed by `uv`. First, synchronize your dependencies:

```fish
cd ml
uv sync
```

### 1. Start MLflow Tracking Server

Run the MLflow server locally using SQLite as the backend store:

```fish
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

### 2. Run the ML Pipeline

You can run individual components defined in the `MLproject`.
**Note 1:** `prepare_data` requires the Lakehouse services (MinIO, Nessie, Trino) to be running.
**Note 2:** You MUST specify `--experiment-name` at the CLI to avoid experiment ID mismatch errors. Open a new terminal to run these steps:

```fish
cd ml

# 1. Prepare data (Fetches from Lakehouse)
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"; uv run mlflow run . -e prepare_data --env-manager local

# 2. Run hyperparameter tuning
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"; uv run mlflow run . -e tune --experiment-name "walmart-sales-tuning" --env-manager local

# 3. Train final baseline & Register model
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"; uv run mlflow run . -e train --experiment-name "walmart-sales-baseline" --env-manager local
```

---

## Integration Points

| Integration | Direction | Description |
|---|---|---|
| **Data Lakehouse (Trino)** | ← Reads | `prepare_data.py` pulls Gold Layer features from Iceberg tables. |
| **`shared/params.yaml`** | ← Reads | Reads `cutoff_date` for splitting and base fallback parameters. |
| **`shared/data_raw/`** | ← Reads | Pulls the Kaggle `test.csv` to flag unseen base rows. |
| **`shared/data/processed/`** | → Writes | Drops `.parquet` files for training and validation splits. |
| **`shared/models/`** | → Writes | Serialises `lgbm_baseline.pkl` here. |
| **MLflow Server** | → Writes | Logs experiments and registers the `@champion` model. |

---

## Related README Files

| Link | Coverage |
|---|---|
| [../shared/README.md](../shared/README.md) | Shared variables and artifact storage behavior. |
| [../data_platform/dbt/README.md](../data_platform/dbt/README.md) | How the Gold Layer tables are formulated in Iceberg. |
| [../backend/README.md](../backend/README.md) | How the API loads the `@champion` model for serving. |
| [../README.md](../README.md) | Root project overview and architecture. |
