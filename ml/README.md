# Machine Learning Core (MLflow Project)

This directory contains the core machine learning logic for sales forecasting, packaged as an MLflow Project.

## Structure

```text
ml/
├── MLproject                 # MLflow project definition
├── pyproject.toml            # ML-specific dependencies (uv)
├── uv.lock                   # uv lockfile
├── processing/               # Data validation & processing logic
│   ├── __init__.py
│   └── validator.py
├── scripts/                  # MLflow entry points scripts
│   ├── __init__.py
│   ├── prepare_data.py       # Data preparation script
│   ├── train.py              # Model training script
│   └── tune.py               # Hyperparameter tuning script
├── tuning/                   # Hyperparameter tuning logic
│   ├── __init__.py
│   └── objective.py          # Optuna objective with nested runs
└── utils/                    # Shared utilities
    ├── __init__.py
    └── mlflow_utils.py       # MLflow setup & config helpers
```

## Environment Management

This project uses **uv** for dependency management. To ensure your local environment matches the project requirements:

```bash
cd ml
uv sync
```

## MLflow Tracking Server

This project uses **PostgreSQL** as the backend store and **MinIO (S3)** as the artifact store. 

### 1. Prerequisites
Ensure you have configured your `.env` file in the root directory (based on `example.env`) and that your terminal has these variables loaded, or the MLflow server will not be able to connect to MinIO.

### 1. Start the Server
Run the following command in the `ml/` directory to start the tracking server:

```bash
uv run mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "postgresql+psycopg2://postgres:changeme@localhost:5432/postgres" \
  --default-artifact-root "s3://mlflow/artifacts"
```
*(Note: Ensure the `mlflow` (or `postgres`) database and the `mlflow` bucket in MinIO exist before starting).*

## Running the Project

### Using MLflow CLI (Recommended)

#### Option A: Running from the Root directory
```powershell
# REQUIRED (PowerShell): CLI needs this environment to find the server
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"

uv --directory ml run mlflow run . -e train --experiment-name walmart-sales-baseline --env-manager local
```

#### Option B: Running from the `ml/` directory
```powershell
# Windows (PowerShell)
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"

cd ml
uv run mlflow run . -e tune --experiment-name walmart-sales-tuning --env-manager local
```

*Note: `--env-manager local` is used to leverage the environment already set up by uv.*

### Running Scripts Directly

You can also run the scripts as Python modules from the `ml/` directory:

```bash
cd ml
python -m scripts.train
```
