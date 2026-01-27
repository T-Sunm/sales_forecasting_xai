from airflow.datasets import Dataset

# Logical Dataset URIs for Airflow orchestration
# Note: Spark uses s3a:// protocol, but Dataset URI is just an identifier
URI_CURATED_SALES = "s3://datalake/curated/sales"
URI_CURATED_WEATHER = "s3://datalake/curated/weather"
URI_FEATURE_STORE = "s3://datalake/feature_store/sales_forecast"

# Dataset Objects for Orchestration
DS_CURATED_SALES = Dataset(URI_CURATED_SALES)
DS_CURATED_WEATHER = Dataset(URI_CURATED_WEATHER)
DS_FEATURE_STORE = Dataset(URI_FEATURE_STORE)
