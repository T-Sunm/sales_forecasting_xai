# Machine Learning Engineering Core

> **Scope** This directory houses the core Machine Learning pipelines for the sales forecasting system. It leverages LightGBM for regression and Optuna for hyperparameter tuning. These scripts are designed to be orchestrated by DVC from the shared directory or run locally via `uv`.

---

## Overview

The ml module standardizes the model building lifecycle. The flow covers extracting processed features from the Iceberg Gold layer via Trino. Operations include splitting data based on time cutoffs, discovering hyperparameters via Optuna, and producing a finalized LightGBM model artifact with pkl extension.

---

## Directory Layout

```
ml/
├── pyproject.toml              ← Python dependencies (managed by uv)
├── processing/
│   └── validator.py            ← Feature column definitions and target mapping
├── scripts/
│   ├── prepare_data.py         ← Trino extraction and Train/Valid/Test splitting
│   ├── train.py                ← Final model training script
│   └── tune.py                 ← Optuna hyperparameter study script
└── tuning/
    └── objective.py            ← Optuna trial evaluation logic
```

---

## Responsibilities

1. **prepare_data** Connects to the Trino gateway to join Gold layer tables including fact_sales_item_daily and fact_store_weather_daily. This provides over 74 features for the training session and applies Kaggle test ID masking.
2. **tune** Reads dataset files and executes an Optuna study to optimize model parameters.
3. **train** Fits a final LightGBM Regressor based on best parameters.

---

## Configuration

This module relies on environment variables loaded from the root `.env`.

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

The Python environment is strictly managed by `uv`.

```fish
cd ml
uv sync
```

### Option A: Via DVC (Recommended)

DVC orchestrates high-level stages. Note that `prepare_data` now requires the Lakehouse services (MinIO, Nessie, Trino) to be running.

```fish
cd ../shared
dvc repro
```

### Option B: Direct Execution

You can run individual scripts directly using `uv run`.

```fish
cd ml

# 1. Prepare data (Fetches from Lakehouse)
uv run scripts/prepare_data.py

# 2. Run hyperparameter tuning
uv run scripts/tune.py

# 3. Train final model
uv run scripts/train.py --best-params outputs/best_params.json
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

---

## Related README Files

| Link | Coverage |
|---|---|
| [../shared/README.md](../shared/README.md) | DVC pipeline locking and artifact storage behavior. |
| [../data_platform/dbt/README.md](../data_platform/dbt/README.md) | How the Gold Layer tables are formulated in Iceberg. |
| [../backend/README.md](../backend/README.md) | How the API loads the model for serving. |
| [../README.md](../README.md) | Root project overview and architecture. |

