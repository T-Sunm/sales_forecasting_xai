# Sales Forecasting XAI

## Overview

This repository hosts an end-to-end Walmart sales forecasting system combining a modern Lakehouse architecture with a comprehensive MLOps pipeline. The data platform leverages MinIO as the object storage, Apache Iceberg as the open table format, and Nessie as the catalog versioning. Apache Spark processes data ingestion and feature engineering while dbt and Apache Airflow orchestrate SQL transformations and pipeline scheduling. On the Machine Learning side, the pipeline utilizes DVC for data and artifact versioning, MLflow for experiment tracking, and Optuna for automated hyperparameter tuning. Model predictions and Explainable AI (XAI) insights are served through a high-performance Trino query engine, a FastAPI backend, and a Streamlit interactive dashboard.

## Architecture

![Architecture Diagram](./assets/main.png)

## Repository Map

| Module | Role | Link |
|---|---|---|
| `data_platform/` | Lakehouse infrastructure and orchestration pipelines | [Read](./data_platform/README.md) |
| ├── `infra/` | Docker services and network overview | [Read](./data_platform/infra/README.md) |
| │ ├── `spark_minio/` | Spark cluster, MinIO storage, Iceberg configuration | [Read](./data_platform/infra/spark_minio/README.md) |
| │ ├── `nessie/` | Nessie Catalog for Iceberg | [Read](./data_platform/infra/nessie/README.md) |
| │ ├── `trino/` | Trino Query Engine for analytical serving | [Read](./data_platform/infra/trino/README.md) |
| │ ├── `airflow/` | Airflow orchestration (CeleryExecutor) | [Read](./data_platform/infra/airflow/README.md) |
| │ └── `postgres/` | Nessie and Airflow metadata database | [Read](./data_platform/infra/postgres/README.md) |
| ├── `dbt/` | dbt data models | [Read](./data_platform/dbt/README.md) |
| └── `spark/` | PySpark ingestion & specific feature engineering jobs | [Read](./data_platform/spark/README.md) |
| `ml/` | ML training algorithms, Optuna tuning, MLflow tracking | [Read](./ml/README.md) |
| `shared/` | DVC pipeline execution & shared artifacts | [Read](./shared/README.md) |
| `backend/` | FastAPI REST API for predictions and XAI | [Read](./backend/README.md) |
| `frontend/` | Streamlit interactive UI & dashboards | [Read](./frontend/README.md) |

## Project Structure

```text
sales_forecasting_xai/
├── backend/                # FastAPI backend & routes (prediction, xai)
├── data_platform/          # Core lakehouse, orchestration & data transformation
│   ├── dbt/                # SQL transformation models
│   ├── infra/              # Containerized infrastructure services
│   ├── pipelines/          # Airflow pipelines placeholder
│   └── spark/              # Spark job definitions & configs
├── frontend/               # Streamlit application
├── ml/                     # ML modeling, training (LightGBM) & Optuna pipelines
└── shared/                 # DVC stages (dvc.yaml), parameters (params.yaml)
```

## Quickstart Local Development

### Assumptions & Prerequisites

*   **Docker & Docker Compose** are installed and running.
*   **Python 3.10+** and the **`uv`** package manager are installed.
*   **DVC** is available globally or within your python environment.
*   The following ports must be free to use: `9000`, `9001` (MinIO), `5432` (Postgres), `19120` (Nessie), `7077` (Spark), `8080` (Airflow), `5000` (MLflow), `8000` (FastAPI), `8501` (Streamlit).
*   An external Docker network named `data_platform_net` must be created before launching the services.

### Minimum Happy Path

The container cluster must be started in a specific sequence to satisfy runtime dependencies starting from PostgreSQL, Nessie, Spark, MinIO, Trino, and then Airflow. Once the infrastructure is ready, the DVC pipeline runs data transformations and trains the machine learning models. Afterward, the API backend and the application UI are initialized.

```fish
# 1. Create the shared external network
docker network create data_platform_net

# 2. Start PostgreSQL (Metadata Store)
cd data_platform/infra/postgres; and docker compose up -d

# 3. Start Nessie Catalog
cd ../nessie; and docker compose up -d

# 4. Start Spark Cluster and MinIO
cd ../spark_minio; and docker compose up -d

# 5. Start Trino Query Engine
cd ../trino; and docker compose up -d

# 6. Initialize and start Apache Airflow
cd ../airflow; and docker compose up airflow-init; and docker compose up -d

# 6. Start the MLflow tracking server locally in a separate terminal context
# cd ../../../../
# set -x MLFLOW_TRACKING_URI http://127.0.0.1:5000
# set -x MLFLOW_S3_ENDPOINT_URL http://localhost:9000
# mlflow server --host 0.0.0.0 --port 5000

# 7. Run the DVC pipeline to execute data preparation, tuning, and training stages
cd ../../../shared; and dvc repro

# 8. Start the FastAPI backend system in a separate terminal context
cd ../backend; and uv run uvicorn src.api.main:app --reload --port 8000

# 9. Start the Streamlit visualization application in a separate terminal context
cd ../frontend; and uv run streamlit run src/app.py
```

## Environment Variables

The project predominantly utilizes an `.env` file located exclusively at the root directory. Below are the key environment variables identified from configuration files that should be documented:

| Variable | Description / Role | Expected Location |
|---|---|---|
| `MINIO_ACCESS_KEY` | Username / Access Key for the MinIO root user | `.env` |
| `MINIO_SECRET_KEY` | Password / Secret Key for the MinIO root user | `.env` |
| `MINIO_ENDPOINT` | Full HTTP endpoint URL to reach MinIO (e.g., `http://localhost:9000`) | `.env` |
| `POSTGRES_USER` | Master username for the central Postgres instance | `.env` |
| `POSTGRES_PASSWORD` | Master password for Postgres authentication | `.env` |
| `POSTGRES_DB` | Default global database initialization name | `.env` |
| `POSTGRES_HOST` | Database host string address | `.env` |
| `POSTGRES_PORT` | Database connection port (Default: `5432`) | `.env` |
| `MLFLOW_TRACKING_URI` | URI to log metrics and MLflow run configurations (e.g., `http://127.0.0.1:5000`) | `.env` |
| `MLFLOW_S3_ENDPOINT_URL` | MinIO override destination endpoint for MLflow artifact storage | `.env` |
| `AWS_ACCESS_KEY_ID` | S3-compatibility key for MLflow to connect to MinIO correctly | `.env` |
| `AWS_SECRET_ACCESS_KEY` | S3-compatibility secret for MLflow to connect to MinIO correctly | `.env` |
| `PGADMIN_DEFAULT_EMAIL` | Default login email for the pgAdmin database UI | docker-compose / `.env` (TODO: Verify value to expose) |
| `PGADMIN_DEFAULT_PASSWORD` | Default login password for the pgAdmin database UI | docker-compose / `.env` (TODO: Verify value to expose) |

> **TODO:** If DVC enforces remote registry storage (e.g., AWS S3 or MinIO), appropriate external tracking credentials will optionally need to be included. Furthermore, if Streamlit or FastAPI requires external binding reference parameters (e.g. backend host IP lookup), these variables need to be formally supplemented to `.env`.

## Decision Log

*   **Why Iceberg, Nessie, and MinIO?** Apache Iceberg facilitates core ACID transactions and schema evolutions on large data lakes. Nessie provides catalog branching features to avoid storage duplication. MinIO replicates an underlying high-performance, S3-compatible local object storage format.
*   **Why Trino and Spark?** Apache Spark provides a robust compute engine for transformation workloads due to built-in fault tolerance. Trino enables low-latency analytical queries on the Gold layer for model training and API serving.
*   **Why DVC, MLflow, and Optuna?** DVC tracks specific iterations for large datasets and exported model objects. MLflow centralizes training metrics metadata dashboards. Optuna automates hyperparameter sweeping alongside nested MLflow metric aggregations.
*   **Why FastAPI and Streamlit?** FastAPI provides asynchronous endpoints to serve project predictions and Explainable AI structures. Streamlit enables the creation of real-time frontend charts for visual delivery.

## What to Read Next

We invite you to onboard module-by-module to get comprehensive knowledge of the architectural breakdown:

1.  **[Data Platform Infrastructure](./data_platform/infra/README.md)** – Understand the initial Docker-based orchestration services integration and networking rules.
2.  **[Airflow Orchestration Capabilities](./data_platform/infra/airflow/README.md)** – Delve into the core DAG execution layout and new asset-driven data scheduling practices.
3.  **[Spark & MinIO Data Lake Integration](./data_platform/infra/spark_minio/README.md)** – Dive heavily into scaling Apache Iceberg concepts inside our custom cluster nodes alongside Apache Nessie table definitions.
4.  **[dbt Implementations](./data_platform/dbt/README.md)** – Learn data warehouse modeling layers utilizing modern analytical SQL mappings conventions.
5.  **[Shared Data Tracking & DVC Usage](./shared/README.md)** – Ascertain reproducibility techniques manipulating our parameter mappings via Data Version Control.
6.  **[Machine Learning Engineering Core](./ml/README.md)** – Assess our deep integrations encompassing `LightGBM` regressions arrays tracked carefully amongst `Optuna` studies inside `MLflow` loops.
