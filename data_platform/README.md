# Data Platform (Lakehouse Architecture)

This project implements a modern data platform using **Spark + MinIO (Data Lake)** and **dbt + PostgreSQL (Data Warehouse)**.

## 🏗️ Architecture Overview

```
Raw CSV → MinIO (Bronze) → Spark Staging → MinIO (Silver) → Spark Intermediate → MinIO (Gold) → PostgreSQL → dbt Marts
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Lake** | MinIO | Object storage for Bronze/Silver/Gold layers |
| **Processing** | Spark | Distributed data transformation |
| **Data Warehouse** | PostgreSQL | Serving layer for analytics |
| **Transformation** | dbt | Star schema modeling |
| **Orchestration** | Airflow | Workflow scheduling |

## 📂 Project Structure

```text
data_platform/
├── infra/                              # Infrastructure
│   ├── postgres/
│   │   ├── docker-compose.yml
│   │   └── scripts/
│   │       └── load_from_minio.py      # Load Gold → PostgreSQL
│   └── spark_minio/
│       ├── docker-compose.yml
│       ├── Dockerfile
│       └── scripts/
│           ├── load_raw_data.py        # Raw CSV → Bronze
│           └── load_holidays.py        # Holidays → Bronze
│
├── datalake/                           # Data Lake metadata
│   ├── schemas/                        # Schema definitions
│   └── docs/                           # Data dictionary
│
├── spark/                              # Spark ETL jobs
│   ├── src/                            # Reusable libraries
│   ├── jobs/
│   │   ├── staging/                    # Bronze → Silver
│   │   └── intermediate/               # Silver → Gold
│   └── configs/                        # Environment configs
│
├── pipelines/                          # Orchestration
│   └── airflow/
│       └── dags/
│
├── dbt/
│   ├── sales_forecasting/              # [LEGACY] Pure dbt implementation
│   └── sales_forecasting_warehouse/    # [NEW] Lakehouse + dbt
│       ├── models/
│       │   ├── sources/                # Sources from Gold layer
│       │   └── marts/
│       │       └── star_schema/        # Fact & Dimension tables
│       ├── seeds/
│       ├── tests/
│       └── macros/
│
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## 🚀 Quick Start

### 1. Setup Shared Network

The infrastructure components communicate via a shared Docker network named `data_platform_net`. You must create this network once before starting the stack:

```powershell
docker network create data_platform_net
```

### 2. Start Infrastructure

With the shared network created, you can now start the components in any order.

```powershell
# 1. Start Spark + MinIO
cd infra/spark_minio
docker-compose up -d

# 2. Start PostgreSQL (Data Warehouse)
cd infra/postgres
docker-compose up -d

# 3. Start Airflow (Orchestrator)
cd infra/airflow
docker-compose up -d
```

### 3. Load Raw Data to MinIO

```powershell
# Load raw CSV files to Bronze layer
uv run python infra/spark_minio/scripts/load_raw_data.py

# Load holidays data
uv run python infra/spark_minio/scripts/load_holidays.py
```

### 3. Run Spark Jobs

```powershell
# Staging: Bronze → Silver
spark-submit spark/jobs/staging/sales_staging.py

# Intermediate: Silver → Gold
spark-submit spark/jobs/intermediate/sales_features.py
```

### 4. Load to PostgreSQL

```powershell
# Load Gold data to PostgreSQL
uv run python infra/postgres/scripts/load_from_minio.py
```

### 6. Run dbt (Manual)

```powershell
cd dbt/sales_forecasting_warehouse
uv run dbt run
```

## 📊 Data Layers

### Bronze Layer (Raw)
- **Storage**: MinIO `bronze/` bucket
- **Format**: Parquet
- **Content**: Raw data from CSV files, no transformations
- **Source**: `infra/spark_minio/scripts/load_raw_data.py`

### Silver Layer (Staged)
- **Storage**: MinIO `silver/` bucket
- **Format**: Parquet
- **Content**: Cleaned and staged data
- **Transformations**: 
  - Column renaming
  - Type casting
  - Basic data cleaning
  - Missing value handling
- **Source**: `spark/jobs/staging/`

### Gold Layer (Features)
- **Storage**: MinIO `gold/` bucket
- **Format**: Parquet
- **Content**: Feature-engineered data ready for ML/Analytics
- **Transformations**:
  - Lag features
  - Rolling window aggregations
  - EWMA calculations
  - Store/Item context features
  - Date features
  - Weather integration
- **Source**: `spark/jobs/intermediate/`

### Marts Layer (Star Schema)
- **Storage**: PostgreSQL
- **Format**: Tables
- **Content**: Dimensional model for BI tools
- **Transformations**: Fact & Dimension tables
- **Source**: `dbt/sales_forecasting_warehouse/models/marts/`

## 🛠️ Technical Refinements (Airflow 3 + Spark 3.12)

Recently, the platform was upgraded to ensure consistency and stability across the stack:

### 1. Python Environment Alignment
- **Uniform Version:** Both Airflow (Driver) and Spark (Executors) now use **Python 3.12**.
- **Spark Configuration:** Explicitly pointing Spark to the correct Python executable via:
  - `spark.pyspark.python: "/usr/bin/python3.12"` (Executors)
  - `spark.pyspark.driver.python: "python3"` (Driver - auto-resolves in Airflow PATH)
- **Distutils Fix:** Python 3.12 removes `distutils`. We've patched this by upgrading `setuptools` and `wheel` in the Spark Dockerfile to provide a compatibility layer.

### 2. Airflow 3 (Asset-Aware) Orchestration
- **From Datasets to Assets:** Migrated from `airflow.datasets.Dataset` to `airflow.sdk.Asset`.
- **Event Lookup:** In Airflow 3, `triggering_asset_events` lookup should be done by iterating and checking the `.uri` or using the URI string as a key to avoid `unhashable dict` errors when the asset carries metadata.
- **Outlet Metadata:** Using Asset objects as keys in `context["outlet_events"]` to properly attach metadata (like `run_date`) for downstream consumers.

### 3. Integrated Connectivity
- **postgres_dw Connection:** Added as an environment variable in `infra/airflow/docker-compose.yaml` to ensure Cosmos (dbt) and Spark jobs consistently point to the Data Warehouse container (`postgres_container`).
  - `AIRFLOW_CONN_POSTGRES_DW`: `postgresql://postgres:changeme@postgres_container:5432/sales_forecasting`

## 🔄 Migration Notes

### Legacy Implementation (dbt-only)
The original implementation in `dbt/sales_forecasting/` used pure dbt for all transformations:
- Staging layer: SQL-based cleaning
- Intermediate layer: SQL window functions for features
- Marts layer: Star schema

**Limitations**:
- Limited scalability with large datasets
- Complex window functions in SQL
- All processing in PostgreSQL

### New Implementation (Lakehouse)
The new implementation in `dbt/sales_forecasting_warehouse/` separates concerns:
- **Spark**: Heavy-lifting transformations (staging + intermediate)
- **dbt**: Final star schema modeling
- **MinIO**: Scalable object storage
- **PostgreSQL**: Serving layer only

**Benefits**:
- Horizontally scalable with Spark
- Better separation of concerns
- Reusable Spark libraries
- Easier to test and maintain

## 📚 Resources

- [dbt Documentation](https://docs.getdbt.com/)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [MinIO Documentation](https://min.io/docs/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
