# Sales Forecasting XAI

## Overview

This repository hosts an end-to-end Walmart sales forecasting system combining a modern Lakehouse architecture with a comprehensive MLOps pipeline. The data platform leverages **MinIO** as the object storage, **Apache Iceberg** as the open table format, and **Nessie** for Git-like catalog versioning. Data ingestion and heavy feature engineering (such as Exponentially Weighted Moving Averages) are processed by **Apache Spark**, while **dbt** and **Apache Airflow** orchestrate resilient SQL transformations and pipeline scheduling. On the Machine Learning side, the pipeline utilizes **DVC** for data and artifact versioning, **MLflow** for experiment tracking, and **Optuna** for automated hyperparameter tuning. Finally, model predictions and Explainable AI (XAI) insights are served through a high-performance **FastAPI** backend and visualized in a **Streamlit** interactive dashboard.

## Architecture

![Architecture Diagram](./assets/main.png)

## Repository Map

| Module | Role | Link |
|---|---|---|
| `data_platform/` | Lakehouse infrastructure & orchestration pipelines | [Read](./data_platform/README.md) |
| ├── `infra/` | Docker services & network overview | [Read](./data_platform/infra/README.md) |
| │ ├── `spark_minio/` | Spark cluster, MinIO storage, Iceberg configuration | [Read](./data_platform/infra/spark_minio/README.md) |
| │ ├── `nessie/` | Nessie Catalog for Iceberg | [Read](./data_platform/infra/nessie/README.md) |
| │ ├── `airflow/` | Airflow orchestration (CeleryExecutor) | [Read](./data_platform/infra/airflow/README.md) |
| │ └── `postgres/` | Airflow metadata DB & Nessie JDBC backend | [Read](./data_platform/infra/postgres/README.md) |
| ├── `dbt/` | dbt data models (Lakehouse vs Warehouse) | [Read](./data_platform/dbt/README.md) |
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

## Quickstart (Local/Dev)

### Assumptions & Prerequisites

*   **Docker & Docker Compose** are installed and running.
*   **Python 3.10+** and the **`uv`** package manager are installed.
*   **DVC** is available globally or within your python environment.
*   The following ports must be free to use: `9000`, `9001` (MinIO), `5432` (Postgres), `19120` (Nessie), `7077` (Spark), `8080` (Airflow), `5000` (MLflow), `8000` (FastAPI), `8501` (Streamlit).
*   An external Docker network named `data_platform_net` must be created before launching the services.

### Minimum Happy Path

The container cluster must be started in a specific sequence to satisfy runtime dependencies: **PostgreSQL → Nessie → Spark & MinIO → Airflow**. Once the infrastructure is completely ready, we trigger the DVC pipeline to run data transformations and train the machine learning models. Afterward, we start the API backend and the app UI.

```fish
# 1. Create the shared external network
docker network create data_platform_net

# 2. Start PostgreSQL (Backend for Airflow & Nessie)
cd data_platform/infra/postgres; and docker compose up -d

# 3. Start Nessie Catalog
cd ../nessie; and docker compose up -d

# 4. Start Spark Cluster and MinIO
cd ../spark_minio; and docker compose up -d

# 5. Initialize and start Apache Airflow
cd ../airflow; and docker compose up airflow-init; and docker compose up -d

# 6. Start the MLflow tracking server locally (run in a separate terminal context)
# cd ../../../../
# set -x MLFLOW_TRACKING_URI http://127.0.0.1:5000
# set -x MLFLOW_S3_ENDPOINT_URL http://localhost:9000
# mlflow server --host 0.0.0.0 --port 5000

# 7. Run the DVC pipeline to execute data preparation, tuning, and training stages
cd ../../../shared; and dvc repro

# 8. Start the FastAPI backend system (run in a separate terminal context)
cd ../backend; and uv run uvicorn src.api.main:app --reload --port 8000

# 9. Start the Streamlit visualization application (run in a separate terminal context)
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

*   **Why Iceberg / Nessie / MinIO?** Apache Iceberg facilitates core ACID transactions and safe schema evolutions on large data lakes; Nessie injects Git-like catalog branching features avoiding storage duplication; MinIO dynamically replicates an underlying high-performance, S3-compatible local testing object storage format without needing AWS.
*   **Why connect dbt with a Spark Thrift target?** Providing a Thrift JDBC/ODBC endpoint intrinsically allows dbt to directly schedule and submit advanced distributed SQL statement transformations across our data platform workers without integrating a discrete data warehouse processing engine.
*   **Why `applyInPandas` for EWMA logic?** Forecasting traits like Exponentially Weighted Moving Averages inherently require rigorous sequential ordering mechanisms that translate poorly in pure Spark SQL context; employing PySpark's `applyInPandas` executes rapid and inherently safe pandas vectorization bounds across mapped data partitions naturally.
*   **Why DVC + MLflow + Optuna?** DVC robustly locks down specific iterations for massive datasets and exported model objects over time; MLflow tightly centralizes complex training metrics metadata dashboards dynamically; Optuna effortlessly automates resilient and statistically profound hyperparameter sweeping alongside nested MLflow metric aggregations intuitively.
*   **Why FastAPI + Streamlit?** FastAPI ships with unparalleled async endpoints ideally suited to wrap rapid ML models predictions alongside dense backend Explainable AI structures; paired with Streamlit, it enables code-agnostic creation of compelling real-time frontend charts to elegantly simplify the visual delivery processes.

## What to Read Next

We invite you to onboard module-by-module to get comprehensive knowledge of the architectural breakdown:

1.  **[Data Platform Infrastructure](./data_platform/infra/README.md)** – Understand the initial Docker-based orchestration services integration and networking rules.
2.  **[Airflow Orchestration Capabilities](./data_platform/infra/airflow/README.md)** – Delve into the core DAG execution layout and new asset-driven data scheduling practices.
3.  **[Spark & MinIO Data Lake Integration](./data_platform/infra/spark_minio/README.md)** – Dive heavily into scaling Apache Iceberg concepts inside our custom cluster nodes alongside Apache Nessie table definitions.
4.  **[dbt Implementations](./data_platform/dbt/README.md)** – Learn data warehouse modeling layers utilizing modern analytical SQL mappings conventions.
5.  **[Shared Data Tracking & DVC Usage](./shared/README.md)** – Ascertain reproducibility techniques manipulating our parameter mappings via Data Version Control.
6.  **[Machine Learning Engineering Core](./ml/README.md)** – Assess our deep integrations encompassing `LightGBM` regressions arrays tracked carefully amongst `Optuna` studies inside `MLflow` loops.
