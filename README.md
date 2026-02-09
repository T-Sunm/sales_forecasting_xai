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

## Architecture & Tech Stack

### 1. Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Data Lake** | MinIO | S3-compatible storage: Data Lake, DVC remote, MLflow artifacts. |
| **Data Processing** | Apache Spark | Distributed ETL and feature engineering. |
| **Data Warehouse** | PostgreSQL | DWH for marts + MLflow backend store. |
| **Transformation** | DBT | SQL-based data transformations on PostgreSQL. |
| **Orchestration** | Apache Airflow | Data pipeline scheduling and monitoring. |
| **Pipeline Versioning** | DVC | Reproducibility, versioning data and ML pipelines. |
| **ML Platform** | MLflow | Experiment tracking and Model Registry. |
| **Hyperparameter Tuning** | Optuna | Automated hyperparameter optimization. |
| **ML Core** | LightGBM, Scikit-learn | Gradient boosting models for forecasting. |
| **Explainability** | SHAP | Model interpretation and feature importance. |
| **Backend API** | FastAPI | Model serving and inference endpoints. |
| **Frontend** | Streamlit | Dashboard and visualization interface. |

### 2. Data Architecture

The data flows through a multi-layer architecture to ensure quality and usability:

- **Raw Layer (Bronze):** Original data ingested from external sources (Kaggle), stored in MinIO.
- **Staging Layer (Silver):** Cleaned and standardized data, processed by Spark, stored in MinIO/Postgres.
- **Intermediate Layer:** Data with business logic applied, joins between tables, prepared for final aggregation.
- **Marts Layer (Gold):** Final, aggregated tables optimized for analytics and ML modeling (e.g., `fact_sales`, `dim_store`).

### 3. Serving Architecture

The system uses a decoupled client-server architecture for scalability:

- **Model Registry:** Trained models are versioned and stored in MLflow.
- **Backend (FastAPI):**
    - Loads the best model from MLflow at startup.
    - Exposes REST endpoints for `prediction`, `health`, and `xai` (explanations).
    - Handles data validation using Pydantic schemas.
- **Frontend (Streamlit):**
    - Consumes the Backend APIs to fetch predictions and SHAP values.
    - Renders interactive charts and natural language insights for end-users.

### 4. MLOps Workflow Integration

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
│   │   ├── mlflow/                 # MLflow Tracking Server
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

## Quick Start 🚀

To get started with the project, follow these steps:

### 1. Global Setup

Create the environment file used by all services:

```bash
cp example.env .env
# Open .env and adjust variables as needed
```

### 2. Module Setup & Execution

Each module manages its own dependencies and execution logic:

#### ML Core
- **What:** DVC pipeline (prepare → tune → train → evaluate), MLflow tracking, Optuna tuning.
- **Docs:** [`ml/README.md`](./ml/README.md)

#### Data Platform
- **What:** Spark ETL, dbt models, Airflow DAGs.
- **Docs:** [`data_platform/README.md`](./data_platform/README.md)

#### Backend API
- **What:** FastAPI serving + `/prediction` + `/xai` endpoints.
- **Docs:** [`backend/README.md`](./backend/README.md)

#### Frontend Dashboard
- **What:** Streamlit UI for predictions + SHAP visualizations.
- **Docs:** [`frontend/README.md`](./frontend/README.md)

#### Shared Resources
- **What:** Jupyter notebooks and common utilities.
- **Docs:** [`shared/README.md`](./shared/README.md)

## Contact

**📧 moonlig73@gmail.com**
