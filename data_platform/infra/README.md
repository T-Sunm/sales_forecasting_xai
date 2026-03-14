# Data Platform — Infrastructure

> **Scope:** This directory owns every containerised service that the data platform depends on. No application code lives here — only Docker Compose stacks, Dockerfiles, and the configuration that wires them together.

---

## Overview

The infra layer is composed of **four independent Compose stacks** that must be started in a specific order because of hard runtime dependencies:

```
postgres → nessie → spark_minio → trino → airflow
```

All four stacks share a single external Docker network (`data_platform_net`) that must be created manually before any stack is launched. The network boundary is the only coupling between stacks; each stack is otherwise self-contained.

---

## Responsibilities

| Sub-directory | What it owns |
|---|---|
| [`postgres/`](./postgres/) | Central PostgreSQL 16 instance and pgAdmin UI. Serves as Airflow metadata database and Nessie JDBC version-store backend. |
| [`nessie/`](./nessie/) | Project Nessie catalog server — provides branching for Apache Iceberg tables stored in MinIO. |
| [`spark_minio/`](./spark_minio/) | Apache Spark cluster and MinIO object storage. Iceberg runtime and Nessie extensions are integrated into the custom Spark image. |
| [`trino/`](./trino/) | Trino analytical engine for executing high-performance SQL queries over Iceberg tables. |
| [`airflow/`](./airflow/) | Apache Airflow 3.1.6 with CeleryExecutor and internal core services. The custom image bundles dbt-spark and Astronomer Cosmos. |

---

## Services & Ports

### `postgres/` stack

| Container | Image | Host Port | Role |
|---|---|---|---|
| `postgres_container` | `postgres:16-alpine` | `5432` | SQL database for Nessie JDBC backend and Airflow metadata |
| `pgadmin_container` | `dpage/pgadmin4:9.11` | `5050` | pgAdmin web UI |

> Source: `postgres/docker-compose.yml`

### `nessie/` stack

| Container | Image | Host Port | Role |
|---|---|---|---|
| `nessie_container` | `ghcr.io/projectnessie/nessie:0.107.2-java` | `19120` | Nessie REST API + UI |

> Source: `nessie/docker-compose.yml`.

### `spark_minio/` stack

| Container | Image | Host Port | Role |
|---|---|---|---|
| `spark-master` | custom (`apache/spark:3.5.3-…`) | `8081` (UI), `7077` (master), `4040` (app UI) | Spark master |
| `spark-worker-1` | same | `8082` | Spark worker (2 cores, 4 GB) |
| `spark-thrift` | same | `10001` | HiveServer2 / Spark Thrift for dbt |
| `minio` | `minio/minio:RELEASE.2025-04-22T22-12-26Z` | `9000` (API), `9001` (console) | S3-compatible object storage |

> Source: `spark_minio/docker-compose.yml`.

### `airflow/` stack

| Container | Image | Host Port | Role |
|---|---|---|---|
| `airflow-apiserver` | custom (base `apache/airflow:3.1.6`) | `8080` | Airflow web UI & REST API |
| `airflow-scheduler` | same | — | DAG scheduling |
| `airflow-dag-processor` | same | — | DAG file parsing |
| `airflow-worker` | same | — | Celery task execution |
| `airflow-triggerer` | same | — | Deferred task execution |
| `redis` | `redis:7.2-bookworm` | `6379` (internal only) | Celery message broker |
| `postgres` (airflow-internal) | `postgres:16` | — (internal only) | Airflow metadata DB |
| `flower` *(optional profile)* | same | `5555` | Celery Flower monitoring |

> Source: `airflow/docker-compose.yaml` lines 104–357.

---

## Configuration

### Shared Network (required before any stack)

```fish
docker network create data_platform_net
```

All four stacks reference this network as `external: true` / `name: data_platform_net`.

### `postgres/` — Environment Variables

Variables are read from the root `.env` (`../../.env` relative to `infra/`), or from `POSTGRES_*` defaults baked into the Compose file.

| Variable | Default (from compose) | Description |
|---|---|---|
| `POSTGRES_USER` | `postgres` | Superuser login |
| `POSTGRES_PASSWORD` | `changeme` | Superuser password |
| `PGADMIN_DEFAULT_EMAIL` | `pgadmin4@pgadmin.org` | pgAdmin login e-mail |
| `PGADMIN_DEFAULT_PASSWORD` | `admin` | pgAdmin login password |
| `PGADMIN_PORT` | `5050` | Host port for pgAdmin |

> Source: `postgres/docker-compose.yml` lines 7–32.

### `airflow/` — Environment Variables

Create `airflow/.env` from `airflow/example.env`:

```fish
cp data_platform/infra/airflow/example.env data_platform/infra/airflow/.env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `AIRFLOW_UID` | **Yes** (Linux) | `50000` | UID for volume ownership |
| `AIRFLOW__API_AUTH__JWT_SECRET` | **Yes** | `1` | JWT secret for Airflow 3.x API. **Use a random string in any non-trivial environment.** |

Additional variables are hardcoded in the compose `environment` block and are **not** overridable via `.env` without editing the file:

| Variable | Value (compose) | Notes |
|---|---|---|
| `AIRFLOW__CORE__EXECUTOR` | `CeleryExecutor` | Fixed — do not change without rebuilding |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | `postgresql+psycopg2://airflow:airflow@postgres/airflow` | Airflow internal DB |
| `AIRFLOW__CELERY__BROKER_URL` | `redis://:@redis:6379/0` | Celery broker |
| `S3_ENDPOINT` | `http://minio:9000` | MinIO reachable from within the compose network |
| `DBT_PROFILES_DIR` | `/opt/airflow/dags/dbt/sales_forecasting` | Points dbt to the mounted profiles |

> Source: `airflow/docker-compose.yaml` lines 56–83 and `airflow/.env`.

### Volume Mounts (Airflow → other stacks)

The Airflow containers mount files from sibling stacks at **runtime**. These paths are relative to the `docker-compose.yaml` location:

| Host path | Container path | Purpose |
|---|---|---|
| `../../dbt/` | `/opt/airflow/dags/dbt` | dbt project files accessible to Cosmos |
| `../../spark/jobs/` | `/opt/spark/jobs` | PySpark job scripts submitted via `SparkSubmitOperator` |
| `../../spark/configs/` | `/opt/spark/configs` | Spark Python config module |
| `../spark_minio/spark_conf/spark-defaults.conf` | `/opt/spark/conf/spark-defaults.conf` (read-only) | Shared Spark configuration (Iceberg, Nessie, S3A) |

> Source: `airflow/docker-compose.yaml` lines 84–92.

---

## How to Run (Local)

> **Prerequisites:** Docker Engine running, `docker network create data_platform_net` executed, `airflow/.env` created from `example.env`.

```fish
# Step 1 — PostgreSQL (must be healthy before Nessie and Airflow)
cd data_platform/infra/postgres
docker compose up -d

# Step 2 — Nessie (requires postgres_container on data_platform_net)
cd ../nessie
docker compose up -d

# Step 3 — Spark and MinIO
cd ../spark_minio
docker compose up -d

# Step 4 — Trino
cd ../trino
docker compose up -d

# Step 5 — Airflow
cd ../airflow
docker compose up airflow-init
docker compose up -d
```

### Verify all services are healthy

```fish
# Postgres
docker exec postgres_container pg_isready -U postgres

# Nessie
curl -s http://localhost:19120/api/v1/trees | python3 -m json.tool

# Spark Master UI
curl -s http://localhost:8081 | head -5

# MinIO health
curl -s http://localhost:9000/minio/health/live

# Airflow API
curl -s http://localhost:8080/api/v2/version
```

### Add the Spark connection in Airflow

The DAGs reference `conn_id="spark_default"`. After Airflow is running, register it once via the CLI (run inside the `airflow-apiserver` container or any Airflow container):

```fish
docker exec -it (docker ps -qf "name=airflow-apiserver") \
  airflow connections add spark_default \
    --conn-type spark \
    --conn-host spark://spark-master \
    --conn-port 7077
```

> **TODO:** Confirm exact `conn-host` format expected by `apache-airflow-providers-apache-spark`. Check `SparkSubmitOperator` docs for the `spark_default` connection schema.

---

## Integration Points

```
┌──────────────────────┬────────────────────────────────────────────────────────┐
│ From                 │ To                                                     │
├──────────────────────┼────────────────────────────────────────────────────────┤
│ Airflow DAGs         │ Spark Master (spark://spark-master:7077) via           │
│ (SparkSubmitOperator)│   SparkSubmitOperator + conn_id=spark_default          │
├──────────────────────┼────────────────────────────────────────────────────────┤
│ Airflow DAGs         │ dbt via Cosmos DbtTaskGroup; profiles.yml points       │
│ (Cosmos)             │   to Spark Thrift at spark-thrift:10001                │
├──────────────────────┼────────────────────────────────────────────────────────┤
│ Spark jobs           │ MinIO (s3a://datalake/) for all data reads/writes      │
├──────────────────────┼────────────────────────────────────────────────────────┤
│ Spark jobs           │ Nessie (:19120) as SparkCatalog for Iceberg tables     │
├──────────────────────┼────────────────────────────────────────────────────────┤
│ Nessie               │ postgres_container:5432/nessie for version store       │
├──────────────────────┼────────────────────────────────────────────────────────┤
│ Asset/Dataset events │ DS_RAW_PARQUET_READY → producer_ewma_features DAG     │
│ (datasets.py)        │ DS_EWMA_FEATURES_READY → consumer_lakehouse_pipeline  │
│                      │ DS_LAKEHOUSE_MART_READY → downstream consumers        │
└──────────────────────┴────────────────────────────────────────────────────────┘
```

> Source: `airflow/dags/datasets.py`, `airflow/dags/producers/staging_pipeline.py`, `airflow/dags/producers/sales_pipeline.py`, `airflow/dags/consumers/feature_engineering.py`.

---

## DAG Inventory

| `dag_id` | Schedule / Trigger | Layer | Tool |
|---|---|---|---|
| `bootstrap_nessie_namespaces` | Manual (`schedule=None`) | Infra setup | pyhive → Spark Thrift |
| `producer_ingest_raw` | `@daily` | Ingestion | SparkSubmitOperator |
| `producer_ewma_features` | `DS_RAW_PARQUET_READY` (Asset) | Intermediate | SparkSubmitOperator |
| `consumer_lakehouse_pipeline` | `DS_RAW_PARQUET_READY` + `DS_EWMA_FEATURES_READY` | Staging → Mart | Cosmos DbtTaskGroup |

> Source: `airflow/dags/producers/`, `airflow/dags/consumers/`, `airflow/dags/bootstrap_nessie_namespaces.py`.

---

## Airflow Custom Image

The `airflow/Dockerfile` extends `apache/airflow:3.1.6` with:

- **System packages:** `openjdk-17-jre-headless`, `git`, `libsasl2-dev` (required for PyHive/Thrift).
- **JARs** (downloaded to `/opt/airflow/jars/` at build time):
  - `hadoop-aws-3.3.4.jar`
  - `aws-java-sdk-bundle-1.12.262.jar`
  - `postgresql-42.7.0.jar`
  - `iceberg-spark-runtime-3.5_2.12-1.10.1.jar`
  - `nessie-spark-extensions-3.5_2.12-0.107.2.jar`
- **dbt** installed in an isolated venv at `/opt/airflow/dbt_venv/` (`dbt-spark[PyHive]`).
- **Python packages** from `requirements.txt`: `astronomer-cosmos==1.12.1`, `apache-airflow-providers-apache-spark`, `psycopg2-binary`.
- **PySpark 3.5.3**, `pandas`, `pyarrow` installed without Airflow constraints (to match the Spark cluster).

> Source: `airflow/Dockerfile`.

---

## Spark Custom Image

The `spark_minio/Dockerfile` extends `apache/spark:3.5.3-scala2.12-java17-python3-ubuntu` with:

- **Python 3.12** (from deadsnakes PPA) + `pandas`, `pyarrow`.
- Same set of JARs as the Airflow image (downloaded to `/opt/spark/jars/`).

> Source: `spark_minio/Dockerfile`.

---

## Related README Files

| Link | Coverage |
|---|---|
| [../README.md](../README.md) | Data Platform overview and startup order |
| [./spark_minio/README.md](./spark_minio/README.md) | Spark cluster, MinIO, Iceberg config detail |
| [./nessie/README.md](./nessie/README.md) | Nessie catalog, JDBC backend, namespaces |
| [./airflow/README.md](./airflow/README.md) | Airflow DAG deep-dive, Cosmos config |
| [./postgres/README.md](./postgres/README.md) | PostgreSQL schema, init scripts |
| [../../dbt/README.md](../../dbt/README.md) | dbt model layers and profiles |
| [../../spark/README.md](../../spark/README.md) | PySpark job descriptions |
| [../../../README.md](../../../README.md) | Root project overview |
