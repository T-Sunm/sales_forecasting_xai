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
├── experiments/                    # 🔄 Experiment configs (thay ml/configs/)
│   ├── base.yaml                   # Base config extends by others
│   ├── exp_lightgbm_v1.yaml
│   ├── exp_xgboost_v1.yaml
│   └── README.md                   # Hướng dẫn tạo experiment mới
│
├── ml/                             # Core ML code
│   ├── __init__.py
│   │
│   ├── processing/                 # 🆕 SHARED: Train + Inference đều dùng
│   │   ├── __init__.py
│   │   ├── transformers.py         # Feature transformers (sklearn compatible)
│   │   ├── feature_pipeline.py     # End-to-end feature pipeline
│   │   └── validator.py            # Pandera/Pydantic schema validation
│   │
│   ├── data/                       # Data loading
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── loader.py               # Load từ MinIO/local
│   │
│   ├── models/                     # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base class
│   │   ├── lightgbm_model.py
│   │   ├── xgboost_model.py
│   │   └── factory.py              # Model factory by name
│   │
│   ├── training/                   # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py              # Train + MLflow logging
│   │   └── cross_validator.py
│   │
│   ├── tuning/                     # Optuna integration
│   │   ├── __init__.py
│   │   ├── objective.py            # Optuna objective
│   │   ├── search_spaces.py        # Search space definitions
│   │   └── tuner.py                # Study wrapper
│   │
│   ├── evaluation/                 # Metrics & XAI
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── explainer.py            # SHAP wrapper
│   │
│   ├── serving/                    # 🆕 Inference wrapper
│   │   ├── __init__.py
│   │   ├── predictor.py            # Load from MLflow Registry + predict
│   │   └── model_loader.py         # MLflow model URI loader
│   │
│   └── utils/
│       ├── __init__.py
│       ├── mlflow_utils.py
│       ├── config_loader.py        # 🆕 Load & merge configs
│       └── logger.py
│
├── scripts/                        # DVC stage entry points
│   ├── prepare_data.py
│   ├── tune.py                     # Output: outputs/tuning/best_params.json
│   ├── train.py                    # Input: best_params.json (optional override)
│   ├── evaluate.py
│   └── register_model.py           # Push to MLflow Registry
│
├── outputs/                        # DVC tracked outputs
│   ├── tuning/
│   │   └── best_params.json        # 🆕 Kết quả Optuna -> Train reads this
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
│   │   └── mlflow/                 # 🆕 MLflow server config
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
uv sync
```

**2. Setup Infrastructure (Docker)**

Navigate to `data_platform/infra/` and start services (Spark, MinIO, Postgres, MLflow).

**3. Run MLOps Pipeline**

```bash
# Prepare data, tune hyperparameters, and train model
dvc repro

# Or run individual steps
dvc repro tune
dvc repro train
```

## Contact

**📧 moonlig73@gmail.com**
