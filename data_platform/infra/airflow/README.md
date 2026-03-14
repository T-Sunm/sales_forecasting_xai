# Airflow Orchestration

> **Scope:** This directory owns the Apache Airflow 3.1.6 stack — its Docker image, Compose services, DAGs, environment configuration, and all pipeline orchestration logic for the data platform.

---

## Overview

Airflow is the **sole orchestrator** for the data platform. It schedules and executes every pipeline stage: raw data ingestion, EWMA feature engineering, and the dbt-based Lakehouse transformation. It runs as a Celery-based cluster (api-server → scheduler → dag-processor → worker → triggerer) backed by an internal Redis broker and an internal Postgres metadata database.

The pipeline is designed around Airflow 3.x **Asset-driven scheduling** (formerly Datasets). DAGs do not poll on time — instead, they react to logical data checkpoints called *Assets*. A DAG only runs when its upstream Asset has been produced by a prior DAG. This means the entire Lakehouse pipeline is **event-driven, not cron-driven** (with the single exception of `producer_ingest_raw`, which runs `@daily` to kick off the chain).

---

## Directory Contents

```
airflow/
├── Dockerfile             ← Custom image: Airflow 3.1.6 + Java 17 + JARs + dbt venv
├── docker-compose.yaml    ← Full Celery cluster definition
├── .env                   ← Required: AIRFLOW_UID + JWT_SECRET (see Configuration)
├── example.env            ← Template — copy to .env before first run
├── requirements.txt       ← Python packages added to image (Cosmos, Spark provider, psycopg2)
├── config/
│   └── airflow.cfg        ← Airflow configuration (mounted at /opt/airflow/config/)
├── dags/
│   ├── datasets.py        ← Central Asset/Dataset definitions (3 checkpoints)
│   ├── bootstrap_nessie_namespaces.py  ← One-time infra setup DAG
│   ├── producers/
│   │   ├── staging_pipeline.py    ← DAG: producer_ingest_raw
│   │   └── sales_pipeline.py      ← DAG: producer_ewma_features
│   └── consumers/
│       └── feature_engineering.py ← DAG: consumer_lakehouse_pipeline
├── logs/                  ← Airflow task logs (mounted volume)
└── plugins/               ← Custom Airflow plugins (empty)
```

---

## Services & Ports

All services are defined in `docker-compose.yaml` (357 lines).

| Container | Command | Host Port | Description |
|---|---|---|---|
| `airflow-apiserver` | `api-server` | **`8080`** | Web UI + REST API v2 |
| `airflow-scheduler` | `scheduler` | — | Parses and schedules DAG runs |
| `airflow-dag-processor` | `dag-processor` | — | Dedicated DAG file processor (Airflow 3.x) |
| `airflow-worker` | `celery worker` | — | Executes tasks via Celery; also acts as Spark driver host |
| `airflow-triggerer` | `triggerer` | — | Handles deferred/async tasks |
| `airflow-init` | one-shot | — | DB migration + admin user creation |
| `redis` | — | `6379` (internal) | Celery message broker |
| `postgres` (internal) | — | — (internal) | Airflow metadata DB only |
| `flower` *(profile: flower)* | `celery flower` | **`5555`** | Celery task monitor (opt-in) |

> **Note:** The internal `postgres` service (inside `airflow/docker-compose.yaml`) is **separate** from `postgres_container` in `infra/postgres/`. Do not confuse the two. The internal one handles Airflow metadata only.

---

## Custom Docker Image

**Base:** `apache/airflow:3.1.6`

The `Dockerfile` augments the base image with the following layers (in order):

### System packages (`root`)
- `openjdk-17-jre-headless` — required for Spark driver execution inside the worker.
- `git` — required by some dbt/Cosmos internals.
- `libsasl2-dev` — required for PyHive SASL authentication to Spark Thrift Server.

### JARs (downloaded to `/opt/airflow/jars/`)
Downloaded from Maven Central at image build time:

| JAR | Version | Purpose |
|---|---|---|
| `hadoop-aws` | 3.3.4 | S3A filesystem for MinIO |
| `aws-java-sdk-bundle` | 1.12.262 | AWS SDK (S3A dependency) |
| `postgresql` | 42.7.0 | JDBC driver for Postgres writes |
| `iceberg-spark-runtime-3.5_2.12` | 1.10.1 | Iceberg table format |
| `nessie-spark-extensions-3.5_2.12` | 0.107.2 | Nessie catalog for Iceberg |

These JARs are passed on-the-fly to `SparkSubmitOperator` via the `jars=` parameter so the Spark driver running inside the Airflow worker container can reach MinIO and Nessie.

### Python virtual environment (`airflow` user)
A separate venv at `/opt/airflow/dbt_venv/` installs `dbt-spark[PyHive]`. This venv is the Python interpreter used by `bootstrap_nessie_namespaces.py` and is the `dbt_executable_path` for Cosmos.

### Main Python packages (`requirements.txt`)
| Package | Version | Purpose |
|---|---|---|
| `astronomer-cosmos` | 1.12.1 | Cosmos: dbt-as-Airflow-tasks |
| `apache-airflow-providers-apache-spark` | (latest) | `SparkSubmitOperator` |
| `psycopg2-binary` | (latest) | Postgres access from DAG tasks |

Additionally, `pyspark==3.5.3`, `pandas`, `pyarrow` are installed **without** Airflow constraints to match the Spark cluster version.

---

## Configuration

### Step 1 — Create `.env`

```fish
cp data_platform/infra/airflow/example.env data_platform/infra/airflow/.env
```

| Variable | Required | Default in `.env` | Notes |
|---|---|---|---|
| `AIRFLOW_UID` | **Yes** (Linux/Mac) | `50000` | Must match your host user UID to avoid permission errors on mounted volumes. On Linux: `id -u`. |
| `AIRFLOW__API_AUTH__JWT_SECRET` | **Yes** | `1` | JWT signing key for Airflow 3.x. Replace with a strong random string in any shared/team environment. |

### Step 2 — Environment Variables in Compose

These variables are **hardcoded** in `docker-compose.yaml` and are not overridable via `.env` without editing the file:

| Variable | Value | Notes |
|---|---|---|
| `AIRFLOW__CORE__EXECUTOR` | `CeleryExecutor` | Fixed. |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | `postgresql+psycopg2://airflow:airflow@postgres/airflow` | Points to internal Airflow postgres. |
| `AIRFLOW__CELERY__RESULT_BACKEND` | `db+postgresql://airflow:airflow@postgres/airflow` | Celery result store. |
| `AIRFLOW__CELERY__BROKER_URL` | `redis://:@redis:6379/0` | Redis broker (internal). |
| `AIRFLOW__CORE__EXECUTION_API_SERVER_URL` | `http://airflow-apiserver:8080/execution/` | Worker → API server for task execution. |
| `S3_ENDPOINT` | `http://minio:9000` | MinIO endpoint reachable inside the compose network |
| `DBT_PROFILES_DIR` | `/opt/airflow/dags/dbt/sales_forecasting` | The mounted dbt project profiles directory |
| `AIRFLOW__CORE__LOAD_EXAMPLES` | `true` | Disable in production by changing to `false`. |

### Volume Mounts (critical for DAG execution)

| Host path (relative to `docker-compose.yaml`) | Container path | Purpose |
|---|---|---|
| `./dags` | `/opt/airflow/dags` | All DAG Python files |
| `./logs` | `/opt/airflow/logs` | Task log storage |
| `./config` | `/opt/airflow/config` | `airflow.cfg` |
| `./plugins` | `/opt/airflow/plugins` | Custom plugins |
| `../../dbt` | `/opt/airflow/dags/dbt` | dbt project files (consumed by Cosmos) |
| `../../spark/jobs` | `/opt/spark/jobs` | PySpark scripts submitted via `SparkSubmitOperator` |
| `../../spark/configs` | `/opt/spark/configs` | Spark Python config module |
| `../spark_minio/spark_conf/spark-defaults.conf` | `/opt/spark/conf/spark-defaults.conf` (read-only) | Shared Spark config (Iceberg, Nessie, S3A) |

---

## Asset-Driven Scheduling Architecture

All pipeline inter-DAG dependencies are expressed via **Airflow Assets** (Airflow 3.x). The central definition file is `dags/datasets.py`.

### Asset Definitions

```python
# dags/datasets.py
URI_RAW_PARQUET_READY    = "s3://datalake/staging/parquet"
URI_EWMA_FEATURES_READY  = "s3://datalake/intermediate/int_sales_with_ewma"
URI_LAKEHOUSE_MART_READY = "nessie://analytics/mart_sales_forecast"
```

### Pipeline Flow
**TODO:** 

> `consumer_lakehouse_pipeline` uses `schedule=[DS_RAW_PARQUET_READY, DS_EWMA_FEATURES_READY]` — it waits for **both** assets to be produced before triggering.

---

## DAG Reference

### `bootstrap_nessie_namespaces`

| Field | Value |
|---|---|
| File | `dags/bootstrap_nessie_namespaces.py` |
| `dag_id` | `bootstrap_nessie_namespaces` |
| Schedule | `None` (manual trigger only) |
| Tags | `infra` |
| Operator | `PythonOperator` |
| Task | `create_nessie_namespaces` |
| Retries | 2 |

**What it does:** Runs a PyHive script (via the `dbt_venv` Python) to execute `CREATE NAMESPACE IF NOT EXISTS` for:
- `nessie.default`
- `nessie.analytics`

Connects to `spark-thrift:10001`. Must be triggered **once** before any dbt models run.

---

### `producer_ingest_raw`

| Field | Value |
|---|---|
| File | `dags/producers/staging_pipeline.py` |
| `dag_id` | `producer_ingest_raw` |
| Schedule | `@daily` |
| Tags | `layer:ingestion`, `domain:all` |
| Operator | `SparkSubmitOperator` |
| `conn_id` | `spark_default` |
| Application | `/opt/spark/jobs/staging/ingest_raw_to_parquet.py` |
| `catchup` | `False` |

**What it does:** Reads raw CSV files from MinIO and converts them to Parquet in the staging zone. Emits `DS_RAW_PARQUET_READY` on success.

**JARs passed at runtime:**
- `hadoop-aws-3.3.4.jar`
- `aws-java-sdk-bundle-1.12.262.jar`
- `iceberg-spark-runtime-3.5_2.12-1.10.1.jar`
- `nessie-spark-extensions-3.5_2.12-0.107.2.jar`

**Spark driver config:**
```
spark.submit.deployMode    = client
spark.driver.host          = airflow-worker
spark.driver.bindAddress   = 0.0.0.0
spark.driver.extraClassPath = /opt/airflow/jars/*
```

---

### `producer_ewma_features`

| Field | Value |
|---|---|
| File | `dags/producers/sales_pipeline.py` |
| `dag_id` | `producer_ewma_features` |
| Schedule | `[DS_RAW_PARQUET_READY]` (Asset-triggered) |
| Tags | `layer:intermediate`, `domain:sales`, `tool:spark` |
| Operator | `SparkSubmitOperator` |
| `conn_id` | `spark_default` |
| Application | `/opt/spark/jobs/intermediate/sales_features_pipeline.py` |
| `max_active_runs` | 1 |
| `max_active_tasks` | 1 |

**What it does:** Computes Exponentially Weighted Moving Average (EWMA) sales features using `applyInPandas` — a Spark-only step that cannot be expressed in pure SQL. Emits `DS_EWMA_FEATURES_READY` with metadata (`run_date`, `batch_id`) attached to the Asset event.

---

### `consumer_lakehouse_pipeline`

| Field | Value |
|---|---|
| File | `dags/consumers/feature_engineering.py` |
| `dag_id` | `consumer_lakehouse_pipeline` |
| Schedule | `[DS_RAW_PARQUET_READY, DS_EWMA_FEATURES_READY]` (both required) |
| Tags | `layer:lakehouse`, `domain:all`, `tool:dbt` |
| `catchup` | `False` |

**What it does:** Runs the entire `sales_forecasting_lakehouse` dbt project via Astronomer Cosmos. Tests run after all models complete (`TestBehavior.AFTER_ALL`). On success, emits `DS_LAKEHOUSE_MART_READY` via a downstream `EmptyOperator`.

**Cosmos config:**

| Parameter | Value |
|---|---|
| `project_config` | `ProjectConfig("/opt/airflow/dags/dbt/sales_forecasting_lakehouse")` |
| `profile_name` | `sales_forecasting_lakehouse` |
| `target_name` | `dev` |
| `profiles_yml_filepath` | `/opt/airflow/dags/dbt/sales_forecasting_lakehouse/profiles.yml` |
| `dbt_executable_path` | `/opt/airflow/dbt_venv/bin/dbt` |
| `test_behavior` | `AFTER_ALL` |

**dbt output (Iceberg tables in Nessie):**
- `nessie.default.*` — staging and intermediate views
- `nessie.analytics.mart_sales_forecast` — final mart table

---

## How to Run (Local)

### Prerequisites

- `data_platform_net` Docker network exists.
- `postgres_container` (from `infra/postgres/`) and `spark-thrift` (from `infra/spark_minio/`) are running and healthy.
- `airflow/.env` created from `example.env`.

### Start Airflow

```fish
cd data_platform/infra/airflow

# First launch only: migrate DB and create admin user
docker compose up airflow-init
# Wait until airflow-init exits with code 0

# Start all services
docker compose up -d
```

### Verify services are running

```fish
# API server health
curl -s http://localhost:8080/api/v2/version

# Check all containers are up
docker compose ps
```

### Register the Spark connection (one-time setup)

The DAGs use `conn_id="spark_default"`. This connection is not created automatically. Register it via the Airflow REST API or CLI:

```fish
# Via CLI inside any running airflow container
docker exec -it (docker ps -qf "name=airflow-apiserver") \
  airflow connections add spark_default \
    --conn-type spark \
    --conn-host spark-master \
    --conn-port 7077
```

**TODO:** Confirm the correct `conn-type` string expected by `apache-airflow-providers-apache-spark`. Check provider documentation for valid values.

### Bootstrap Nessie namespaces (one-time setup)

After `spark-thrift` is healthy, manually trigger the bootstrap DAG once:

```fish
docker exec -it (docker ps -qf "name=airflow-apiserver") \
  airflow dags trigger bootstrap_nessie_namespaces
```

### Enable optional Celery Flower monitoring

```fish
docker compose --profile flower up -d
# Access at http://localhost:5555
```

### Trigger a full pipeline run manually

```fish
# Trigger the root DAG; downstream DAGs will fire automatically via Asset events
docker exec -it (docker ps -qf "name=airflow-apiserver") \
  airflow dags trigger producer_ingest_raw
```

---

## Integration Points

| Airflow component | Connects to | How |
|---|---|---|
| `SparkSubmitOperator` (all producer DAGs) | `spark-master:7077` | `conn_id=spark_default`; Spark driver runs inside `airflow-worker` |
| Spark driver (inside worker) | `minio:9000` | S3A + JAR bundle; credentials via `spark-defaults.conf` |
| Spark driver (inside worker) | `nessie:19120` | Nessie extensions JAR; config via `spark-defaults.conf` |
| `bootstrap_nessie_namespaces` | `spark-thrift:10001` | PyHive from `dbt_venv`; `CREATE NAMESPACE` statements |
| Cosmos `DbtTaskGroup` | `spark-thrift:10001` | dbt profile `type=spark, method=thrift`; executable from `dbt_venv` |
| Airflow scheduler/worker | `redis:6379` | Celery broker internal to compose |
| Airflow all components | `postgres` internal | Airflow metadata database |

---

## Related README Files

| Link | Coverage |
|---|---|
| [../README.md](../README.md) | Infra overview, startup order, all services |
| [../spark_minio/README.md](../spark_minio/README.md) | Spark cluster detail, MinIO, Iceberg config, `spark-defaults.conf` |
| [../nessie/README.md](../nessie/README.md) | Nessie catalog, namespaces, JDBC backend |
| [../../dbt/README.md](../../dbt/README.md) | dbt model layers and profiles consumed by Cosmos |
| [../../spark/README.md](../../spark/README.md) | PySpark job descriptions submitted by `SparkSubmitOperator` |
| [../../../README.md](../../../README.md) | Root project overview and quickstart |
