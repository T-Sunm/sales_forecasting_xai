# Shared Resources & Data Project

This directory acts as the **Single Source of Truth** for data, models, and shared utilities in the Sales Forecasting XAI project. It is managed independently as a "Data Project" using DVC.

## 📂 Structure

- `data/data_raw/`: Immutable raw data (tracked by DVC). Source of truth for all downstream pipelines.
- `data/processed/`: ML-ready parquet files (output of `ml/prepare_data.py`).
- `data/data_processed/`: Legacy processed data (for backend/frontend compat).
- `models/`: Trained models and feature statistics.
- `notebooks/`: Exploration and prototyping.
- `utils/`: Shared utility functions.

## 🛠️ DVC Setup (MinIO Remote)

We use **MinIO** as the object storage backend for DVC. Follow these steps to set up your environment.

### 1. Prerequisites

Ensure MinIO is running (`http://localhost:9000`) and the `dvc-store` bucket exists.

**Option A: Run script (recommended)**
```bash
cd data_platform
uv run python infra/spark_minio/scripts/create_dvc_bucket.py
```

**Option B: Use MinIO Web Console**
1. Open http://localhost:9001
2. Login: `minioadmin` / `minioadmin`
3. Buckets -> Create Bucket -> Name: `dvc-store`

### 2. Configure DVC Remote

Run the following commands in the `shared/` directory.

**A. Global Config (Commited to Git)**
Defines the remote storage location.

```bash
dvc remote add -d minio s3://dvc-store/sales_forecasting_xai/shared
dvc remote modify minio endpointurl http://localhost:9000
```

**B. Local Config (NOT Commited - Sensitive)**
Sets your local credentials. Do NOT commit these to Git.

```bash
dvc remote modify --local minio access_key_id minioadmin
dvc remote modify --local minio secret_access_key minioadmin
```

> **Note:** The `--local` flag saves settings to `.dvc/config.local`, which is git-ignored by default.

### 3. Usage

**Push Data to Remote:**
```bash
dvc push
```

**Pull Data from Remote (for new clones):**
```bash
dvc pull
```
