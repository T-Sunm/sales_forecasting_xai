try:
    from airflow.sdk import Asset
except ImportError:
    from airflow.datasets import Dataset as Asset

URI_STG_DATA = "s3://datalake/staging/ready"
URI_INTER_SALES = "s3://datalake/intermediate/sales"
URI_INTER_WEATHER = "s3://datalake/intermediate/weather"
URI_MART_FEATURES = "s3://datalake/mart/features"

DS_STG_DATA = Asset(URI_STG_DATA)
DS_INTER_SALES = Asset(URI_INTER_SALES)
DS_INTER_WEATHER = Asset(URI_INTER_WEATHER)
DS_MART_FEATURES = Asset(URI_MART_FEATURES)
