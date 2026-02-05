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
├── .env / example.env              # Environment variables
├── .gitignore                      # Git ignore rules
├── .python-version                 # Python version
├── environment.yml                 # Conda environment
├── README.md                       # Project documentation
│
├── ml/                             # Core ML Module (MLflow Project)
│   ├── MLproject                   # MLflow entry points definition
│   ├── pyproject.toml              # ML dependencies (uv)
│   ├── uv.lock
│   ├── README.md
│   │
│   ├── scripts/                    # MLflow entry points
│   │   ├── prepare_data.py         # Data preparation
│   │   ├── train.py                # Model training
│   │   └── tune.py                 # Hyperparameter tuning (Optuna)
│   │
│   ├── tuning/                     # Optuna tuning logic
│   │   └── objective.py
│   │
│   ├── processing/                 # Data processing
│   │   └── validator.py
│   │
│   ├── utils/                      # Shared utilities
│   │   └── mlflow_utils.py
│   │
│   └── outputs/tuning/             # Tuning results (best_params.json)
│
├── data_platform/                  # Data Engineering Layer
│   ├── README.md
│   ├── pyproject.toml
│   │
│   ├── infra/                      # Infrastructure (Docker)
│   │   ├── airflow/                # Apache Airflow
│   │   │   ├── docker-compose.yaml
│   │   │   ├── Dockerfile
│   │   │   └── dags/
│   │   │       ├── producers/      # Data ingestion DAGs
│   │   │       └── consumers/      # Data consumption DAGs
│   │   │
│   │   ├── spark_minio/            # Spark + MinIO
│   │   │   ├── docker-compose.yml
│   │   │   └── Dockerfile
│   │   │
│   │   ├── postgres/               # PostgreSQL Data Warehouse
│   │   │   └── docker-compose.yml
│   │   │
│   │   └── mlflow/                 # MLflow Tracking Server
│   │
│   ├── spark/                      # Spark Jobs
│   │   ├── configs/
│   │   └── jobs/
│   │       ├── staging/            # Raw -> Staging
│   │       ├── intermediate/       # Staging -> Intermediate
│   │       └── load_to_postgres.py
│   │
│   └── dbt/                        # DBT Transformations
│       ├── sales_forecasting/      # Lakehouse Project
│       │   ├── macros/
│       │   └── models/
│       │       ├── staging/
│       │       ├── intermediate/
│       │       └── marts/
│       │
│       └── sales_forecasting_warehouse/  # PostgreSQL Project
│
├── backend/                        # FastAPI Application
│   ├── pyproject.toml
│   ├── run.py
│   │
│   └── src/
│       ├── api/
│       │   ├── main.py
│       │   └── routers/
│       │       ├── health.py
│       │       ├── models.py
│       │       ├── prediction.py
│       │       └── xai.py
│       │
│       ├── core/                   # Business logic
│       │   ├── model.py
│       │   ├── forecasting.py
│       │   └── xai_explainer.py
│       │
│       └── data_loader/
│
├── frontend/                       # Streamlit Application
│   ├── pyproject.toml
│   │
│   └── src/
│       ├── app.py
│       │
│       ├── components/
│       │   ├── ui_builder/
│       │   ├── ui_predictor/
│       │   └── ui_xai/             # XAI Dashboard
│       │       ├── shap_plots.py
│       │       ├── explainer.py
│       │       └── llm_explainer.py
│       │
│       └── services/
│           └── api_client.py
│
└── shared/                         # Shared Resources
    ├── notebooks/                  # Jupyter Notebooks
    │   └── wallmart_data/          # Walmart sales analysis
    │       ├── 01_preprocessing.ipynb
    │       ├── 02_EDA.ipynb
    │       ├── 03_feature_engineering.ipynb
    │       ├── 04_modelling_*.ipynb
    │       └── 05_explain_model.ipynb
    │
    ├── data/                       # Shared data files
    │   ├── processed/              # ML-ready data (output of prepare_data.py)
    │   │   ├── train.parquet
    │   │   ├── valid.parquet
    │   │   └── test.parquet
    │   └── data_raw/               # Raw Kaggle data
    │
    └── utils/
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
