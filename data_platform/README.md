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

### 1. Start Infrastructure

```powershell
# Start PostgreSQL
cd infra/postgres
docker-compose up -d

# Start Spark + MinIO
cd infra/spark_minio
docker-compose up -d
```

### 2. Load Raw Data to MinIO

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

### 5. Run dbt

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
