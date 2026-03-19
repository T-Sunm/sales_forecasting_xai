# Data Transformations — dbt Core

> **Scope:** This directory manages the Python environment and orchestrates the dbt project responsible for transforming raw sales data into an analytics-ready dimensional model.

---

## Overview

The data platform utilizes dbt Core targeting a local PostgreSQL instance as its execution and storage engine. Raw data is ingested directly from static CSV files via `dbt seed`, standardized in intermediate views, and ultimately materialized as highly optimized fact and dimension tables.

## Python Environment

Dependency management is handled securely and deterministically using `uv`, defined via a centralized `pyproject.toml`.

| Dependency | Purpose |
| --- | --- |
| `dbt-core` | Core transformation engine framework |
| `dbt-postgres` | PostgreSQL-specific adapter for database integration |

---

## How to Run (Local)

### Prerequisites

- The underlying PostgreSQL container from the `infra/postgres/` stack must be active and accessible on port `5432`.
- Install pipeline dependencies using `uv` prior to initiating any models.

```powershell
uv sync
cd sales_forecasting
```

### Execute Pipeline

Invoke the transformation workflows sequentially:

```powershell
# Ingest raw CSV datasets into foundational database tables
uv run dbt seed

# Execute SQL transformation models (staging schemas and analytical marts)
uv run dbt run

# Perform schema validation and execute data quality constraints
uv run dbt test
```

## Model Architecture Framework

- **Seeds (`seeds/`):** Immutable raw CSV data providing foundational sales metrics, calendar parameters, and auxiliary features.
- **Staging (`models/staging/`):** Initial standardisation layer consisting of views performing essential data cleansing, missing value imputation, and explicit structural type casting to enforce consistency.
- **Marts (`models/marts/`):** The terminal dimensional modeling layer consisting of materialized fact tables characterizing target business events (e.g., granular daily sales) alongside contextually joined dimension tables ready for BI visualization and ML consumption.

---

## Related Documentation

| Link | Coverage |
|---|---|
| [../README.md](../README.md) | Root Data Platform architecture overview |
| [../infra/README.md](../infra/README.md) | PostgreSQL database infrastructure provisioning guide |
