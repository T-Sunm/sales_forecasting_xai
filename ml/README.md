# Machine Learning Engineering Core

> **Scope:** This directory encapsulates the core Machine Learning operational pipelines for the sales forecasting system. It leverages LightGBM for robust regression and Optuna for systematic hyperparameter tuning. These scripts are engineered to be orchestrated securely via `uv` and seamlessly integrated with MLflow for continuous model tracking and registry operations.

---

## Overview

The `ml` module standardizes the model lifecycle operations adhering to rigorous MLOps practices. The pipeline covers extracting fully-materialized analytical features directly from the PostgreSQL warehouse (specifically encompassing the dbt `marts` layer). Subsequent operations include deterministic time-based data splitting, distributed hyperparameter optimization via Optuna, and synthesizing a finalized LightGBM predictive artifact registered iteratively into the centralized MLflow repository.

---

## Directory Architecture

```text
ml/
├── pyproject.toml              ← Python dependencies and environment specifications (managed via uv)
├── processing/
│   └── validator.py            ← Feature column definitions and exact target variable mapping constraints
├── scripts/
│   ├── prepare_data.py         ← PostgreSQL data extraction and temporal Train/Valid/Test set partitioning
│   ├── train.py                ← Final model training and MLflow `@champion` alias registration
│   └── tune.py                 ← Optuna hyperparameter study execution and metric evaluation
└── tuning/
    └── objective.py            ← Optuna trial evaluation logic and cross-validation analytical mechanics
```

---

## Core Operational Responsibilities

1. **Data Preparation (`prepare_data.py`):** Initiates connections to the PostgreSQL database to retrieve the dimensional feature tables systematically engineered by dbt. This phase strictly enforces train-test temporal splits based on the designated operational cutoff dates.
2. **Hyperparameter Tuning (`tune.py`):** Iterates over the partitioned datasets and executes an Optuna study to optimize the LightGBM regressor's hyperparameters, efficiently tracking all studies and minimizing the loss function.
3. **Model Training (`train.py`):** Fits the terminal LightGBM Regressor utilizing the optimal parameters empirically discovered. It serializes the model artifact and registers it directly into the active MLflow server, subsequently assigning the authoritative `@champion` alias to expedite backend consumption and scalable model serving.

---

## Environment Configuration

This module derives its operational context from environment variables explicitly declared within the root `.env` repository protocol.

### Data Warehouse Integrations (PostgreSQL)

| Variable | Protocol Definition | Default Fallback |
|---|---|---|
| `POSTGRES_USER` | System User Identity | `postgres` |
| `POSTGRES_PASSWORD` | Secure Authentication | `changeme` |
| `POSTGRES_HOST` | Target Domain / IP | `localhost` |
| `POSTGRES_PORT` | Target Communication Port | `5432` |
| `POSTGRES_DB` | Target Database Schema | `postgres` |

---

## Execution Guide (Local MLOps)

The Python runtime environment is strictly and deterministically provisioned by the `uv` package manager.

```powershell
# Initialise the deterministic run environment
cd ml
uv sync
```

### Direct Script Execution Pipeline

Execute the foundational MLOps pipeline stages sequentially:

```powershell
# 1. Prepare data (Extracts materializations dynamically from PostgreSQL)
uv run python scripts/prepare_data.py

# 2. Run expansive hyperparameter optimization study
uv run python scripts/tune.py

# 3. Train the final model artifact & register synchronously into MLflow
uv run python scripts/train.py --best-params outputs/best_params.json
```

---

## System Integration Points

| Integration Node | Data Flow Direction | Functional Description |
|---|---|---|
| **Data Warehouse (PostgreSQL)** | ← Ingest | `prepare_data.py` retrieves records directly from the materialized dbt analytical tables. |
| **`shared/params.yaml`** | ← Ingest | Reads the deterministic `cutoff_date` for temporal splitting and foundational baseline parameters. |
| **`shared/data/processed/`** | → Persist | Isolates local intermediate `.parquet` format checkpoints for contiguous training logic. |
| **MLflow Registry** | → Persist | Serializes the final model payload algorithm via MLflow tracking and promotes it to production serving endpoints. |

---

## Related Documentation Context

| Location | Coverage Context |
|---|---|
| [../data_platform/dbt/README.md](../data_platform/dbt/README.md) | Outlines how analytical feature tables are computationally constructed and materialized. |
| [../backend/README.md](../backend/README.md) | Details how the API service dynamically locates and hosts the MLflow registered model. |
| [../README.md](../README.md) | Specifies Root system overview, extensive architectural topologies, and infrastructure strategies. |
