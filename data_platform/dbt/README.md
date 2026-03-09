# dbt — Data Transformations

> **Scope** This directory contains the dbt project orchestrating the transformation of raw and engineered data into the analytics-ready layer using Apache Spark.

---

## Overview

The data platform utilizes a single dbt project targeting the Spark compute engine:

| Project | Target engine | Output format | Namespace |
|---|---|---|---|
| [`sales_forecasting_lakehouse`](./sales_forecasting_lakehouse/) | Apache Spark via Thrift | **Apache Iceberg tables** | `nessie.default.*` and `nessie.analytics.*` |

The lakehouse project transforms Parquet files from MinIO into a fully featured star schema and ML-ready features. It stores results as versioned Iceberg tables tracked by Nessie.

---

## Responsibilities

### `sales_forecasting_lakehouse`

Performs the full transformation chain from raw Parquet to analytics mart datasets.

---

## Model Inventory — Lakehouse

### Staging layer (`+materialized: view`, `+schema: default`)

| Model | Source | Key output columns |
|---|---|---|
| `stg_sales` | `parquet.s3a://datalake/staging/parquet/train` | `store_id`, `item_id`, `date`, `units` |
| `stg_weather` | `parquet.s3a://datalake/staging/parquet/weather` | weather metrics per station |
| `stg_key` | `parquet.s3a://datalake/staging/parquet/key` | `store_id → station_id` mapping |
| `stg_holidays` | `parquet.s3a://datalake/staging/parquet/holidays` | `date`, `is_holiday` |
| `stg_blackfriday` | `parquet.s3a://datalake/staging/parquet/blackfriday` | `date`, `is_blackfriday` |

> **Note:** Staging models read **directly from MinIO Parquet** via `parquet.\`s3a://...\`` — there is no dbt source YAML; Spark resolves the path through S3A. Source: `staging/stg_sales.sql`.

### Intermediate layer (`+materialized: view`, `+schema: default`)

| Model | Depends on | Key computation |
|---|---|---|
| `int_active_sales` | `stg_sales` | Filter items with zero lifetime sales; compute `ln(units+1)` → `log_units` |
| `int_sales_with_lags` | `int_active_sales` | Jinja loop: `lag(log_units, k) OVER (PARTITION BY store_id, item_id ORDER BY date)` for k ∈ {1,2,3,4,5,6,7,14,21,28} |
| `int_sales_with_rolling` | `int_sales_with_lags` | Rolling avg/min/max/std at windows 7d, 14d, 28d |
| `int_store_item_aggregates` | `int_sales_with_lags` | Store-level and item-level 7d sum/mean |
| `int_date_features` | `stg_sales`, `stg_holidays`, `stg_blackfriday` | Calendar flags: `is_weekend`, `is_holiday`, `is_blackfriday`, season one-hot |
| `int_weather_features` | `stg_weather`, `stg_key` | Numeric imputation + weather code OHE |

### Mart layer (`+materialized: table`, `+file_format: iceberg`)

| Model | Key purpose |
|---|---|
| `fact_sales_item_daily` | Full ML feature row: joins lags + rolling + EWMA (from Spark Parquet) + aggregates |
| `fact_store_weather_daily` | Store × date weather record with profile key |
| `dim_date` | Calendar dimension (date, weekday, season, holiday, blackfriday flags) |
| `dim_item` | Item dimension (`item_id`) |
| `dim_store` | Store → weather station mapping |
| `dim_weather_profile` | Deduped weather condition profiles (OHE flags) |

> **EWMA join in fact_sales_item_daily** The mart reads EWMA features directly from MinIO Parquet `parquet.s3a://datalake/intermediate/int_sales_with_ewma` because EWMA requires sequential per-partition `applyInPandas` which is not expressible in Spark SQL. This is the only external Parquet reference in the mart layer.

---

## ERD — Lakehouse Mart
**TODO**

---

## Configuration

### `sales_forecasting_lakehouse` Profile (`profiles.yml`)

| Parameter | `dev` target | `local` target |
|---|---|---|
| `type` | `spark` | `spark` |
| `method` | `thrift` | `thrift` |
| `host` | `spark-thrift` (Docker hostname) | `localhost` |
| `port` | `10001` | `10001` |
| `schema` | `analytics` | `analytics` |
| `threads` | 4 | 4 |
| `connect_retries` | 3 | 3 |

> Use target `dev` when running inside Docker (Airflow Cosmos). Use `local` when running dbt CLI from your host machine while Spark Thrift is accessible at `localhost:10001`.

### Custom Macro — `generate_schema_name`

The lakehouse project overrides dbt's default schema naming:

```sql
{% macro generate_schema_name(custom_schema_name, node) -%}
  {%- if custom_schema_name is none -%}
    {{ target.schema | trim }}   {# uses target.schema = 'analytics' #}
  {%- else -%}
    {{ custom_schema_name | trim }}  {# uses the model-level +schema config #}
  {%- endif -%}
{%- endmacro %}
```

This means:
- Models with `+schema: default` → materialise in `nessie.default`
- Models without `+schema` override → materialise in `nessie.analytics` (target schema)

### Python Environment

The unified data transformation layer defines its environment requirements in `pyproject.toml` located at `data_platform/dbt/`.

| Dependency | Version |
|---|---|
| `dbt-core` | `~=1.8.0` |
| `dbt-spark[pyhive]` | `>=1.10.1` |

---

## How to Run (Local)

### Prerequisites

- `spark-thrift` container running and reachable at `localhost:10001`.
- MinIO running with `datalake` bucket containing staging Parquet files (produced by `producer_ingest_raw` DAG).
- EWMA Parquet generated at `s3a://datalake/intermediate/int_sales_with_ewma` (produced by `producer_ewma_features` DAG).
- Python environment set up via `uv`:

```fish
cd data_platform/dbt
uv sync
```

### Run the Lakehouse project

```fish
cd data_platform/dbt/sales_forecasting_lakehouse

# Use 'local' target when running from host (spark-thrift at localhost:10001)
uv run dbt run --target local --profiles-dir .

# Run tests after models complete
uv run dbt test --target local --profiles-dir .

# Run only staging layer
uv run dbt run --select staging.* --target local --profiles-dir .

# Run only marts
uv run dbt run --select marts.* --target local --profiles-dir .
```

---

## Integration Points

| Component | Direction | Mechanism |
|---|---|---|
| MinIO `s3a://datalake/staging/parquet/` | ← reads | Staging models query Parquet directly via Spark SQL |
| MinIO `s3a://datalake/intermediate/int_sales_with_ewma` | ← reads | `fact_sales_item_daily` joins EWMA Parquet directly |
| Spark Thrift Server `:10001` | → executes on | dbt submits all SQL via JDBC Thrift |
| Nessie Catalog | → writes to | Mart Iceberg tables land in `nessie.analytics.*` via Spark catalog |
| Airflow Cosmos | ← triggered by | Cosmos `DbtTaskGroup` wraps each model as an Airflow task |
| FastAPI backend | ← reads | Trino serves requests querying Iceberg directly |

---

## Related README Files

| Link | Coverage |
|---|---|
| [../README.md](../README.md) | Data Platform overview |
| [../infra/README.md](../infra/README.md) | Infra overview, startup order |
| [../infra/spark_minio/README.md](../infra/spark_minio/README.md) | Spark cluster and Iceberg config |
| [../infra/nessie/README.md](../infra/nessie/README.md) | Nessie catalog setup |
| [../infra/airflow/README.md](../infra/airflow/README.md) | Cosmos DAG orchestrating this dbt project |
| [../infra/trino/README.md](../infra/trino/README.md) | Trino query engine setup for downstream reading |
| [../spark/README.md](../spark/README.md) | PySpark jobs that produce Parquet inputs read by `stg_*` and EWMA join |
| [../../../README.md](../../../README.md) | Root project overview |
