# Data Platform

This project implements a streamlined and robust data pipeline utilizing dbt (Data Build Tool) and PostgreSQL to process, transform, and manage sales forecasting data. The architecture is designed to focus on scalability, maintainability, and data quality.

## Architecture Overview

```text
Raw CSV → dbt (Seeds) → PostgreSQL (Staging) → PostgreSQL (Marts)
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Storage & Serving** | PostgreSQL | Primary relational database for storing raw ingested data, interim staging views, and final analytical tables |
| **Data Transformation** | dbt Core | SQL-centric transformation engine responsible for orchestrating the ELT (Extract, Load, Transform) framework and enforcing data contracts |

## 📂 Project Structure

```text
data_platform/
├── infra/                              # Infrastructure deployment configurations
│   └── postgres/
│       └── docker-compose.yml          # PostgreSQL database deployment
│
├── dbt/                                # Transformation pipeline logic
│   └── sales_forecasting/
│       ├── models/
│       │   ├── marts/                  # Analytics-ready fact and dimension tables
│       │   └── staging/                # Data cleaning and standardization models
│       ├── seeds/                      # Raw CSV data ingestion templates
│       ├── tests/                      # Declarative data quality tests
│       └── macros/                     # Reusable SQL macros
│
├── pyproject.toml                      # Python dependencies and environments
├── README.md                           # Documentation
└── uv.lock                             # Dependency lockfile
```

## 🚀 Quick Start

### 1. Initialize Infrastructure

Deploy the foundational PostgreSQL database instance. This component must be operational prior to executing downstream pipeline operations.

```powershell
cd infra/postgres
docker-compose up -d
```

### 2. Execute Data Pipeline

Invoke the dbt pipeline to install necessary packages, ingest the source data, and materialize the transformation models within the database layer.

```powershell
cd dbt/sales_forecasting

# Install registered dbt dependencies
uv run dbt deps

# Materialize raw CSV data into PostgreSQL tables
uv run dbt seed

# Execute the transformation models and build the staging/marts layers
uv run dbt run
```

## 📊 Data Architecture Layers

The architecture employs a disciplined, multi-layered modeling structure, strictly adhering to established data engineering practices to guarantee consistency and tracebility:

### Seeds Layer (Raw Data)
- **Storage Paradigm:** PostgreSQL Tables
- **Content:** Immutable source data directly ingested from static CSV files.
- **Source Location:** `dbt/sales_forecasting/seeds/`

### Staging Layer
- **Storage Paradigm:** PostgreSQL Views
- **Content:** Standardized and cleansed intermediate data. Includes minimal business logic to ensure source fidelity.
- **Transformation Strategy:** 
  - Standardized naming conventions and schema alignment
  - Deterministic type casting
  - Null value handling
- **Source Location:** `dbt/sales_forecasting/models/staging/`

### Marts Layer
- **Storage Paradigm:** PostgreSQL Tables
- **Content:** Refined analytical datasets engineered specifically for machine learning ingestion and business intelligence (BI) consumption. Supported by star-schema principles.
- **Transformation Strategy:**
  - Aggregations and derived feature engineering (e.g., temporal features, lag parameters)
  - Dimensional modeling abstractions
- **Source Location:** `dbt/sales_forecasting/models/marts/`

## 🛡️ Data Quality Validation

The pipeline operationalizes testing mechanisms via dbt to ensure structural integrity and validate business assumptions before data is utilized in forecasting models:
- **Schema Validation:** Declarative constraints dictating referential integrity, absolute uniqueness, and non-null adherence.
- **Custom Assertions:** Configured expectations regarding numerical thresholds and categorical domains.

## 📚 Resources

- [dbt Official Documentation](https://docs.getdbt.com/)
- [PostgreSQL Official Documentation](https://www.postgresql.org/docs/)
