# Trino Serving Layer

Trino is used as the high-performance distributed SQL query engine for the lakehouse serving layer. It allows the backend to query Iceberg tables stored in MinIO via the Nessie catalog without moving data to a relational database.

## Architecture

- **Catalog:** Iceberg (via Nessie)
- **Metadata:** Nessie (`http://nessie_container:19120`)
- **Storage:** MinIO (`http://minio:9000`)
- **Query UI:** `http://localhost:8085`

## Configuration

The configuration is located in `etc/catalog/iceberg.properties`.

## How to Run

1. Ensure the shared network exists:
   ```bash
   docker network create data_platform_net
   ```

2. Start the Trino stack:
   ```bash
   docker compose up -d
   ```

## Integration

The backend connects to Trino using the following parameters:
- **Host** `localhost` or `trino` inside docker network
- **Port** `8085` host / `8080` internal
- **Catalog** `iceberg`
- **Schema** `analytics`
