# Machine Learning Engineering Core

> **Scope:** This directory houses the core Machine Learning pipelines for the sales forecasting system. It leverages `LightGBM` for regression, `Optuna` for hyperparameter tuning, and `MLflow` for experiment tracking. These scripts are designed to be orchestrated by DVC (from the `shared/` directory) or run locally via the `mlflow CLI`.

---

## Overview

The `ml/` module is standardising the model building lifecycle. It defines a formal **MLproject** component. The flow covers extracting analytical marts from PostgreSQL, splitting data based on time cutoffs, discovering optimal hyperparameters via Bayesian optimization (Optuna), and producing a finalized LightGBM model artifact (`.pkl`).

All experiments, parameters, and metrics are meticulously tracked using MLflow, with nested runs for Optuna trials to keep the experiment UI clean and traceable.

---

## Directory Layout

```
ml/
├── MLproject                   ← MLflow project definition (entry points)
├── pyproject.toml              ← Python dependencies (managed by uv)
├── processing/
│   └── validator.py            ← Feature column definitions and target mapping
├── scripts/
│   ├── prepare_data.py         ← DB extraction and Train/Valid/Test splitting
│   ├── train.py                ← Final model training script
│   └── tune.py                 ← Optuna hyperparameter study script
├── tuning/
│   └── objective.py            ← Optuna trial evaluation logic (nested MLflow runs)
└── utils/
    └── mlflow_utils.py         ← Centralised MLflow tracking URI & S3 setup
```

---

## Responsibilities

1. **`prepare_data`**: Connects to the data warehouse (`marts.sales_forecast` JOIN `marts.dim_date`), applies Kaggle test ID masking, handles numeric coercion, and splits the data temporally based on `prepare.cutoff_date` from `params.yaml`.
2. **`tune`**: Reads Parquet files, establishes an `optuna.create_study`, and iteratively suggests parameters across a defined search space (`tuning/objective.py`). Each trial is logged as a *nested* child run under a parent MLflow study. Outputs JSON payload containing the `best_params`.
3. **`train`**: Merges default static hyperparams (from `params.yaml`) with tuned `best_params.json` to fit a final LightGBM Regressor. Automatically logs metrics (`val_rmsle`, `val_mae`) using `mlflow.lightgbm.autolog()` and saves a `.pkl` to `shared/models/`.

---

## Configuration

This module relies on environment variables (usually loaded from the root `.env` via `utils/mlflow_utils.py` and `scripts/prepare_data.py`).

### MLflow Tracking & Artifacts
| Variable | Usage | Default Fallback |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Points to the MLflow tracking server | `http://127.0.0.1:5000` |
| `MLFLOW_S3_ENDPOINT_URL` | Endpoint for MinIO (where MLflow stores artifacts) | None |
| `AWS_ACCESS_KEY_ID` | MinIO access key for artifact upload | `minioadmin` |
| `AWS_SECRET_ACCESS_KEY` | MinIO secret key for artifact upload | `minioadmin` |

### Database Extraction (`prepare_data.py`)
| Variable | Usage | Default |
|---|---|---|
| `POSTGRES_USER` | DB User | `postgres` |
| `POSTGRES_PASSWORD` | DB Password | `changeme` |
| `POSTGRES_HOST` | DB Host | `localhost` |
| `POSTGRES_PORT` | DB Port | `5432` |

> *Database used:* Fixed to `sales_forecasting`.

---

## How to Run (Local)

The Python environment is strictly managed by `uv`.

```fish
cd ml
uv sync
```

### Option A: Via DVC (Recommended)

Because data dependencies, input parameters, and output artifacts are closely tied together, DVC is the intended orchestrator.

```fish
cd ../shared
dvc repro
```

### Option B: Via MLflow CLI

You can run individual components defined in `MLproject`:

```fish
cd ml

# 1. Prepare data
uv run mlflow run . -e prepare_data --env-manager local

# 2. Run hyperparameter tuning
uv run mlflow run . -e tune --env-manager local

# 3. Train final baseline
uv run mlflow run . -e train --env-manager local
```

### Option C: Standalone Scripts

If you need to debug a specific script and control arguments explicitly:

```fish
cd ml
uv run python -m scripts.tune \
  --n-trials 10 \
  --study-name "debug_study" \
  --train-path ../shared/data/processed/train.parquet \
  --valid-path ../shared/data/processed/valid.parquet
```

---

## Integration Points

| Integration | Direction | Description |
|---|---|---|
| **PostgreSQL** | ← Reads | `prepare_data.py` pulls historical joins directly from the DBMS via `sqlalchemy`. |
| **`shared/params.yaml`** | ← Reads | Reads `cutoff_date` for splitting and base fallback parameters for `train.py`. |
| **`shared/data_raw/`** | ← Reads | Pulls the Kaggle `test.csv` to flag unseen store/item rows properly. |
| **`shared/data/processed/`** | → Writes | Drops `.parquet` files for training and validation splits. |
| **`shared/models/`** | → Writes | Serialises `lgbm_baseline.pkl` here. |
| **MLflow Server** | → Writes | Logs tags, parent/child hierarchical runs (Optuna trials), and metric dictionaries (`val_rmsle`). |

---

## Related README Files

| Link | Coverage |
|---|---|
| [../shared/README.md](../shared/README.md) | DVC pipeline locking, param definitions, and DVC artifact storage behavior. |
| [../data_platform/dbt/README.md](../data_platform/dbt/README.md) | How the mart tables queried by `prepare_data.py` are structurally formulated. |
| [../backend/README.md](../backend/README.md) | How the API loads the finalized model (`.pkl`) for serving predictions. |
| [../README.md](../README.md) | Root project overview and architecture. |
