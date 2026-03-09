# Data Platform (Lakehouse Architecture)

This project implements a modern data platform using Spark and MinIO for the Data Lake.

## Architecture Overview

```
Raw CSV → MinIO Bronze → Spark Staging → MinIO Silver → Spark Intermediate → MinIO Gold → dbt Iceberg
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Lake** | MinIO | Object storage for Bronze, Silver, and Gold layers |
| **Processing** | Spark | Distributed data transformation engine |
| **Orchestration** | Airflow | Workflow scheduling and dependency management |
| **Metadata** | PostgreSQL | Metadata store for Nessie and Airflow |

## 📂 Project Structure

```text
data_platform/
├── infra/                              # Infrastructure
│   ├── trino/
│   │   └── docker-compose.yml          # Trino for analytical serving
│   ├── postgres/
│   │   └── docker-compose.yml          # Metadata store
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
│   └── sales_forecasting_lakehouse/    # Lakehouse models using Spark
│       ├── models/
│       │   ├── marts/                  # Analytics ready Iceberg tables
│       │   └── staging/                # Initial cleaning models
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

# 2. Start Infrastructure

![Spark Cluster Dashboard](../assets/spark-cluster-dashboard.jpg)

With the shared network created, you can now start the components in any order.

```powershell
# 1. Start Spark and MinIO
cd infra/spark_minio
docker-compose up -d

# 2. Start PostgreSQL (Metadata Store)
cd infra/postgres
docker-compose up -d

# 3. Start Trino (Serving Engine)
cd infra/trino
docker-compose up -d

# 4. Start Airflow (Orchestrator)
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
# Staging transformation from Bronze to Silver
spark-submit spark/jobs/staging/sales_staging.py

# Intermediate transformation from Silver to Gold
spark-submit spark/jobs/intermediate/sales_features.py
```

### 4. Run dbt (Manual)

```powershell
cd dbt/sales_forecasting_lakehouse
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

### Marts Layer Iceberg Tables
- **Storage** MinIO
- **Format** Apache Iceberg
- **Content** Analytical datasets ready for Machine Learning and BI
- **Engine** dbt and Apache Spark
- **Source** `dbt/sales_forecasting_lakehouse/models/marts/`

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

![Airflow DAGs List](../assets/list_dags.jpg)

- **Event Lookup:** In Airflow 3, `triggering_asset_events` lookup should be done by iterating and checking the `.uri` or using the URI string as a key to avoid `unhashable dict` errors when the asset carries metadata.
- **Outlet Metadata:** Using Asset objects as keys in `context["outlet_events"]` to properly attach metadata (like `run_date`) for downstream consumers.

#### Astronomer Cosmos Integration
The platform utilizes Astronomer Cosmos to render dbt projects directly as Airflow Task Groups, allowing for fine-grained dependency management within the Lakehouse.

![dbt Cosmos DAG](../assets/airflow-dbt_cosmos-dag.jpg)

### 3. Nessie Catalog & Iceberg Namespaces
The architecture implements branching and versioning at the data layer using Apache Nessie.

![Nessie Catalog Namespaces](../assets/nessie-catalog-namespaces.jpg)

- **Internal Connections** Airflow uses the centralized PostgreSQL instance to store task metadata and Nessie catalog versioning information.

##  Resources

- [dbt Documentation](https://docs.getdbt.com/)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [MinIO Documentation](https://min.io/docs/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
