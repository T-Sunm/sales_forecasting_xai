# Data Lake

Thư mục này chứa metadata và contracts cho Data Lake (MinIO).

## Cấu trúc

- `schemas/`: Schema definitions và data contracts (Pandera, JSON Schema, Great Expectations)
- `docs/`: Data dictionary và documentation cho các layer:
  - Bronze: Raw data
  - Silver: Cleaned/staged data (từ Spark staging jobs)
  - Gold: Feature-engineered data (từ Spark intermediate jobs)

## Layers

### Bronze
- Raw data từ CSV files
- Không có transformation
- Format: Parquet
- Bucket: `bronze/`

### Silver
- Data đã được cleaned và staged
- Tương đương với staging layer trong dbt
- Transformations: rename columns, cast types, basic cleaning
- Format: Parquet
- Bucket: `silver/`

### Gold
- Data đã được feature engineering
- Tương đương với intermediate layer trong dbt
- Transformations: lags, rolling windows, EWMA, aggregations
- Format: Parquet
- Bucket: `gold/`
