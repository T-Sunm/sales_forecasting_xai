# Backend API Service

FastAPI-based backend service for Sales Forecasting and XAI explanations.

## Features
- **Prediction API** Serve model predictions using LightGBM.
- **XAI API** Provide SHAP-based explanations for model decisions.
- **Health Check** Monitor service status.

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Start MLflow Tracking Server

The backend loads the model via MLflow Model Registry alias @champion. The MLflow server must be running before starting the backend to avoid connection errors on port 5000.

```bash
# Run from ml relative to this backend folder
cd ../ml
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

MLflow UI will be available at `http://127.0.0.1:5000`. The model registry must have a model named `sales-forecasting-lgbm` with an `@champion` alias. If not, run the training script first:

```bash
# Run from ml relative to this backend folder
cd ../ml
uv run python scripts/train.py
```

### 3. Run Server

```bash
# Using Python script
python run.py

# Or via Uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation will be available at `http://localhost:8000/docs`.

## Data Platform Prerequisites

To serve data through Trino for the Backend, initialize the core services forming the Data Lakehouse layer including storage, metadata catalog, and compute engine.

Based on the structure at `../data_platform/infra`, you must run `docker-compose up` for the following services:

### 1. Storage Layer (MinIO)
Storage for Iceberg Parquet files. Directory name `infra/spark_minio`

```powershell
cd ../data_platform/infra/spark_minio
docker-compose up -d
```

### 2. Metadata Store (PostgreSQL)
Store metadata for the Nessie Catalog. Directory name `infra/postgres`

```powershell
cd ../data_platform/infra/postgres
docker-compose up -d
```

### 3. Iceberg Catalog Nessie
REST Catalog for Iceberg table metadata management. Directory name `infra/nessie`

```powershell
cd ../data_platform/infra/nessie
docker-compose up -d
```

### 4. Analytical Query Engine Trino
SQL engine for high performance analytical queries. Directory name `infra/trino`

```powershell
cd ../data_platform/infra/trino
docker-compose up -d
```
