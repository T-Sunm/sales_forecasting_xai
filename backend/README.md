# Backend API Service

FastAPI-based backend service for Sales Forecasting and XAI explanations.

## Features
- **Prediction API**: Serve model predictions (LightGBM/XGBoost).
- **XAI API**: Provide SHAP-based explanations for model decisions.
- **Health Check**: Monitor service status.

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Server
```bash
# Using Python script
python run.py

# Or via Uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
The API documentation will be available at `http://localhost:8000/docs`.

## Data Platform Prerequisites

To serve data through Trino for the Backend, you need to start the core services that form the Data Lakehouse layer (Storage + Metadata Catalog + Compute Engine).

Based on the structure at `../data_platform/infra`, you must run `docker-compose up` for the following services:

### 1. Storage Layer (MinIO)
Where Parquet files (Iceberg format) are stored. Directory name: `infra/spark_minio`

```powershell
cd ../data_platform/infra/spark_minio
docker-compose up -d
```
*(Note: You can just run `docker-compose up -d minio` if you don't need Spark master/worker running in the background)*

### 2. Database Metadata (PostgreSQL)
Stores metadata (version store) for Nessie Catalog. Directory name: `infra/postgres`

```powershell
cd ../data_platform/infra/postgres
docker-compose up -d
```

### 3. Iceberg Catalog (Nessie)
REST Catalog managing Iceberg tables metadata (connects MinIO and PostgreSQL). Directory name: `infra/nessie`

```powershell
cd ../data_platform/infra/nessie
docker-compose up -d
```

### 4. Query Engine (Trino)
SQL Engine directly querying data, acting as the backend endpoint (port 8085). Directory name: `infra/trino`

```powershell
cd ../data_platform/infra/trino
docker-compose up -d
```
