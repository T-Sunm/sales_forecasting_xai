# Spark & MinIO Data Lake Integration

> **Scope:** This directory manages the custom Apache Spark cluster (master, worker, Thrift server) alongside the local MinIO object storage. It orchestrates the tight coupling of Spark with Apache Iceberg (for open table formats) and Apache Nessie (for dataset versioning).

---

## Overview

This module forms the structural backbone of the Lakehouse architecture. **MinIO** replaces AWS S3 to serve as the local scalable object storage backend. **Apache Spark** serves as the heavy-duty data processing engine, handling both PySpark scripts (for complex manipulations like Pandas UDFs) and Spark SQL queries (via the Thrift server).

Crucially, Spark is heavily extended via `spark-defaults.conf` and downloaded JARs to natively support **Apache Iceberg**. Iceberg relies on **Nessie** as its foundational Catalog, allowing Git-like transactional semantics (`BRANCH`, `TAG`, `MERGE`) across the data lake directly from Spark SQL and dbt.

---

## Responsibilities

*   **MinIO:** Act as the local, S3-compatible backend storage for all raw, staging, and structured Iceberg datasets under the `datalake` bucket.
*   **Spark Master/Worker:** Provide the computational cluster tailored for batch jobs scheduled by Airflow.
*   **Spark Thrift Server:** Act as a HiveServer2/ODBC endpoint (`10001`) to allow external tools (like dbt) to execute native Spark SQL natively within the cluster.
*   **Iceberg/Nessie Extension Loading:** Inject all required runtime `.jar` files into the Spark classpath before startup, mapping S3 endpoints seamlessly.

---

## Services & Ports

Defined in `docker-compose.yml`:

| Container | Image | Host Port | Description / Role |
|---|---|---|---|
| `minio` | `RELEASE.2025-04-22T22-12-26Z` | `9000` (API), `9001` (UI) | S3-compatible object storage server. |
| `spark-master` | Custom build | `7077` (Cluster), `8081` (UI), `4040` | Cluster manager/coordinator. |
| `spark-worker-1` | Custom build | `8082` (UI) | Executor node (Configured for 2 cores, 4GB RAM). |
| `spark-thrift` | Custom build | `10001` (Thrift/JDBC) | External JDBC engine (used primarily by dbt). Runs with 1 executor core. |

> All services connect exclusively through the external Docker network: `data_platform_net`.

---

## Configuration

### Environment Variables

MinIO specific variables defined inside `docker-compose.yml`:
*   `MINIO_ROOT_USER`: `minioadmin` (Default)
*   `MINIO_ROOT_PASSWORD`: `minioadmin` (Default)

Spark containers inherit the following variables dynamically via compose files:
*   `S3_ENDPOINT`: `http://minio:9000` (Used internally to route S3 API traffic).
*   `SPARK_PUBLIC_DNS`: `localhost`
*   `SPARK_WORKER_CORES`: `2` (Worker only)
*   `SPARK_WORKER_MEMORY`: `4G` (Worker only)

### Spark Defaults (`spark-defaults.conf`)

This configuration read-only mount maps into all Spark containers and Airflow workers. It controls the Iceberg mapping behavior rigorously:

*   **Iceberg Injection:** `spark.sql.extensions = org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions`
*   **Nessie Catalog Registration:**
    *   `spark.sql.catalog.nessie` = `org.apache.iceberg.spark.SparkCatalog`
    *   `spark.sql.catalog.nessie.catalog-impl` = `org.apache.iceberg.nessie.NessieCatalog`
    *   `spark.sql.catalog.nessie.uri` = `http://nessie:19120/api/v1` (Nessie's REST API point)
    *   `spark.sql.catalog.nessie.ref` = `main`
    *   `spark.sql.catalog.nessie.warehouse` = `s3a://datalake/iceberg/`
*   **S3A Access Defaults:** Forces Spark paths starting with `s3a://` to route correctly through the S3AFileSystem utilizing the `minioadmin` credentials and path-style access configurations.

---

## Custom Docker Image (`Dockerfile`)

Extends `apache/spark:3.5.3-scala2.12-java17-python3-ubuntu`.
In order to handle specific ML and DataFrame configurations correctly alongside Iceberg, the build script adds:

1.  **Python 3.12 via deadsnakes PPA:** Replaces the default Python interpreter constraints.
2.  **PyPi Packages:** Includes `pandas` and `pyarrow` to allow execution of Pandas UDFs (like EWMA computations).
3.  **Maven JARs (pulled to `/opt/spark/jars/`):**
    *   `hadoop-aws-3.3.4.jar` & `aws-java-sdk-bundle-1.12.262.jar` (S3 interoperability).
    *   `postgresql-42.7.0.jar` (Optional postgres interaction).
    *   `iceberg-spark-runtime-3.5_2.12-1.10.1.jar` (Iceberg table formats).
    *   `nessie-spark-extensions-3.5_2.12-0.107.2.jar` (Nessie branching catalog commands).

---

## Integration Points

*   **dbt (via Airflow Cosmos):** Targets the `spark-thrift` JDBC server on port `10001` directly to execute Iceberg table aggregations (`+file_format: iceberg`).
*   **Airflow `SparkSubmitOperator`:** Uses `spark-master:7077` to launch code. Airflow injects identical JAR files natively into its own driver to process DataFrame writes.
*   **Nessie API:** Every Spark DDL (Create Table, Drop Table) using the Iceberg extension interacts first with the `nessie:19120` REST API to retrieve tree structures and commit logs.

---

## How to Run (Local)

Assuming the `data_platform_net` exists from standard architecture startups:

```fish
cd data_platform/infra/spark_minio

# Start the cluster nodes
docker compose up -d
```

> **Important Bootstrap Step**: Before Spark application code attempts to push data onto MinIO, you must log into MinIO Console (http://localhost:9001) or use the SDK to create the `datalake` bucket manually, as MinIO does not auto-initialize buckets.

To view operational logs:
```fish
docker compose logs -f spark-master
docker compose logs -f spark-thrift
```

---

## Links

| Link | Notes |
|---|---|
| [../README.md](../README.md) | Upstream overview of architecture startup sequences. |
| [../airflow/README.md](../airflow/README.md) | How SparkSubmit executes within the Spark Cluster explicitly. |
| [../nessie/README.md](../nessie/README.md) | Nessie Catalog infrastructure overview. |
| [../../dbt/README.md](../../dbt/README.md) | dbt models mapping using Spark Thrift JDBC. |
| [../../spark/README.md](../../spark/README.md) | Python Job definition codebase executing strictly inside MinIO architecture layers. |
