# Infrastructure

Thư mục này chứa các infrastructure definitions và scripts.

## PostgreSQL

Data Warehouse chính cho serving layer.

### Scripts
- `load_from_minio.py`: Load Gold data từ MinIO vào PostgreSQL

## Spark + MinIO

Distributed processing engine và object storage.

### Scripts
- `load_raw_data.py`: Load raw CSV files vào MinIO (Bronze layer)
- `load_holidays.py`: Load holidays data vào MinIO (Bronze layer)

## Running Infrastructure

```bash
# Start PostgreSQL
cd infra/postgres
docker-compose up -d

# Start Spark + MinIO
cd infra/spark_minio
docker-compose up -d

# Start Airflow
cd infra/airflow
docker-compose up -d
```

## Airflow

Orchestrator cho toàn bộ pipeline.

### Setup `.env` (bắt buộc)

Tạo file `infra/airflow/.env` với nội dung:

```env
AIRFLOW_UID=50000
AIRFLOW__API_AUTH__JWT_SECRET=1
```

> **Note**: `AIRFLOW__API_AUTH__JWT_SECRET` bắt buộc cho Airflow 3.x. Trong production, dùng random string thay vì `1`.
