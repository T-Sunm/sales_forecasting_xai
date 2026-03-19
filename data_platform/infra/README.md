# Data Platform — Infrastructure

> **Scope:** This directory owns the containerised PostgreSQL service that underpins the data transformation pipeline. There is no application code located here — only Docker Compose configurations and relevant provisioning environment variables.

---

## Overview

The infrastructure layer has been purposefully simplified to a unified, independent Docker Compose stack. The PostgreSQL instance operates as the sole data warehouse ecosystem, serving as the repository for raw ingested metrics and the execution engine for the materialized analytics layers orchestrated by dbt. All legacy lakehouse components (Spark, Trino, Airflow, MinIO, Nessie) have been deprecated to streamline operational maintainability.

## Services & Ports

### `postgres/` Stack Structure

| Container | Image | Host Port | Role |
|---|---|---|---|
| `postgres_container` | `postgres:16-alpine` | `5432` | Primary database, executing dbt transformations and serving analytical datasets |

> Source Configuration: `postgres/docker-compose.yml`

---

## Configuration Parameters

### Environment Variables

The PostgreSQL service initializes utilizing natively supported environment variables. Essential configuration parameters include:

| Variable | Default (compose) | Description |
|---|---|---|
| `POSTGRES_USER` | `postgres` | Superuser privileged login identity |
| `POSTGRES_PASSWORD` | `changeme` | Superuser authentication password |
| `POSTGRES_DB` | `postgres` | Default initial database provisioned at startup |

---

## Deployment (Local Execution)

> **Prerequisites:** A functional local Docker Engine installation is strictly required.

### 1. Initialize the Database

Spin up the storage tier. This container must reach a healthy state before attempting to execute upstream data engineering pipelines.

```powershell
cd postgres
docker-compose up -d
```

### 2. Verify Service Integrity

Validate that the database server is actively accepting TCP ingress traffic:

```powershell
docker exec postgres_container pg_isready -U postgres
```

---

## Related Documentation

| Link | Coverage |
|---|---|
| [../README.md](../README.md) | Root Data Platform architecture overview |
| [../dbt/README.md](../dbt/README.md) | Transformation pipeline and dbt configuration guide |
