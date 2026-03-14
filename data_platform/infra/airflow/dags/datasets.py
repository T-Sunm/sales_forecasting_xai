try:
    from airflow.sdk import Asset
except ImportError:
    from airflow.datasets import Dataset as Asset

# URI đại diện cho checkpoint của từng layer trong pipeline
URI_RAW_PARQUET_READY = "s3://datalake/staging/parquet"          # sau ingest_raw_to_parquet
URI_EWMA_FEATURES_READY = "s3://datalake/intermediate/int_sales_with_ewma"  # sau EWMA Spark job
URI_LAKEHOUSE_MART_READY = "nessie://analytics/mart_sales_forecast"          # sau dbt run xong

DS_RAW_PARQUET_READY  = Asset(URI_RAW_PARQUET_READY)
DS_EWMA_FEATURES_READY = Asset(URI_EWMA_FEATURES_READY)
DS_LAKEHOUSE_MART_READY = Asset(URI_LAKEHOUSE_MART_READY)
