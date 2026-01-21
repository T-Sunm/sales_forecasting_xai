# Spark ETL Jobs

Thư mục này chứa các Spark jobs để transform data trong Data Lake.

## Cấu trúc

- `src/`: Reusable libraries và utility functions
- `jobs/`: Entry points cho các Spark jobs
  - `staging/`: Transform Bronze → Silver (cleaning, renaming, type casting)
  - `intermediate/`: Transform Silver → Gold (feature engineering)
- `configs/`: Configuration files cho các môi trường (dev/prod)

## Data Flow

```
Bronze (MinIO) → Spark Staging → Silver (MinIO) → Spark Intermediate → Gold (MinIO) → PostgreSQL
```

## Staging Jobs
Tương đương với staging layer trong dbt cũ:
- Rename columns
- Cast data types
- Basic data cleaning
- Handle missing values

## Intermediate Jobs
Tương đương với intermediate layer trong dbt cũ:
- Lag features
- Rolling window aggregations
- EWMA calculations
- Store/Item context features
- Date features
- Weather data integration

## Running Jobs

```bash
# Example: Run staging job
spark-submit spark/jobs/staging/sales_staging.py

# Example: Run intermediate job
spark-submit spark/jobs/intermediate/sales_features.py
```
