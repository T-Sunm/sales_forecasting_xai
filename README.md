# Sales Forecasting with Explainable AI (XAI)

## Overview

**Sales Forecasting with Explainable AI (XAI)** is a complete end-to-end MLOps system designed to leverage machine learning for store-level sales forecasting. Built using **Python, LightGBM, SHAP, Optuna, DVC, MLflow, Airflow, Spark, and Streamlit**, this project combines robust data engineering pipelines with advanced ML workflows to ensure reproducibility, scalability, and interpretability.

## Key Features

- **Advanced MLOps Pipeline:**
  - **DVC:** Data versioning and pipeline orchestration (Prepare -> Tune -> Train -> Evaluate).
  - **MLflow:** Experiment tracking (metrics, params) and Model Registry.
  - **Optuna:** Hyperparameter optimization with best params handoff to training.

- **Data Engineering Layer:**
  - Uses Spark for big data processing and DBT for data transformation in the warehouse/lakehouse.

- **Explainable AI:**
  - SHAP values integration for transparent model predictions.

## Project Structure

```bash
sales_forecasting_xai/
│
├── .dvc/                           # DVC config (auto-generated)
├── .dvcignore                      # Ignore large model files if needed
├── dvc.yaml                        # DVC Pipeline definition
├── dvc.lock                        # DVC Pipeline lock (auto-generated)
├── params.yaml                     # Default params - DVC reads this
│
├── pyproject.toml                  # 🆕 Root dependency management (UV/Poetry)
├── Makefile                        # 🆕 Shortcuts: make tune, make train, make exp
│
├── data/                           # DVC tracked
│   ├── raw/                        # .dvc files only
│   ├── processed/
│   └── features/
│
├── experiments/                    # 🔄 Experiment configs (replaces ml/configs/)
│   ├── base.yaml                   # Base config extended by others
│   ├── exp_lightgbm_v1.yaml
│   ├── exp_xgboost_v1.yaml
│   └── README.md                   # Guide to creating new experiments
│
├── ml/                             # Core ML (MLflow Project)
│   ├── MLproject                 # MLflow project definition
│   ├── pyproject.toml            # ML-specific dependencies (uv)
│   ├── uv.lock                   # uv lockfile
│   ├── processing/               # Data validation & processing logic
│   │   ├── __init__.py
│   │   └── validator.py
│   ├── scripts/                  # MLflow entry points
│   │   ├── __init__.py
│   │   ├── prepare_data.py       # Data preparation script
│   │   └── train.py              # Model training script
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       └── mlflow_utils.py       # MLflow setup & config helpers
│
├── outputs/                        # DVC tracked outputs
│   ├── tuning/
│   │   └── best_params.json        # 🆕 Optuna results -> Train reads this
│   ├── models/                     # Local backup (MLflow is source of truth)
│   ├── metrics/
│   ├── plots/
│   └── shap/
│
├── data_platform/                  # Infra & Data Engineering
│   ├── infra/
│   │   ├── airflow/
│   │   ├── postgres/
│   │   ├── spark_minio/
│   ├── dbt/
│   ├── spark/
│   └── pipelines/
│
├── backend/                        # Application layer (FastAPI)
│   ├── infra/
│   │   ├── airflow/
│   │   ├── postgres/
│   │   ├── spark_minio/
│   ├── dbt/
│   ├── spark/
│   └── pipelines/
│
├── backend/                        # Application layer (FastAPI)
├── frontend/                       # UI layer (Streamlit)
└── shared/notebooks/               # Data Exploration & Prototyping
```

## MLOps Workflow Integration

1.  **DVC (Data Version Control):**
    -   Manages the DAG (prepare -> tune -> train -> evaluate).
    -   Tracks inputs (data) and outputs (models, metrics).
    -   Ensures reproducibility by locking dependency versions.

2.  **MLflow:**
    -   **Experiment Tracking:** Logs all parameters, metrics, and artifacts during training.
    -   **Model Registry:** Acts as the *Source of Truth* for deployable models.

3.  **Optuna:**
    -   Performs automated hyperparameter tuning.
    -   Outputs `best_params.json` which is automatically picked up by the training stage.

## Quick Start 🚀

**1. Setup Environment**

```bash
# Install dependencies
uv sync

# Setup environment variables
cp example.env .env
# Open .env and adjust variables if needed
```

**2. Setup Infrastructure (Docker)**

Navigate to `data_platform/infra/` and start services (Spark, MinIO, Postgres, MLflow).

**3. Run ML Pipeline**

#### Option A: Run from Root directory
```powershell
# 1. Prepare data
uv --directory ml run python -m scripts.prepare_data

# 2. REQUIRED: Set environment variables for the current terminal session
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"

# 3. Run Training or Tuning via MLflow Project
uv --directory ml run mlflow run . -e train --experiment-name walmart-sales-baseline --env-manager local
uv --directory ml run mlflow run . -e tune --experiment-name walmart-sales-tuning --env-manager local
```

#### Option B: Run from `ml/` directory
```powershell
cd ml
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"

uv run mlflow run . -e train --experiment-name walmart-sales-baseline --env-manager local
```

## Contact

**📧 moonlig73@gmail.com**
