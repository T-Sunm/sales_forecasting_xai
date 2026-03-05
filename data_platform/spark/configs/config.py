import os

# MinIO/S3 Configuration
BUCKET_NAME = "datalake"
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")  # MinIO endpoint
S3_ACCESS_KEY = "minioadmin"
S3_SECRET_KEY = "minioadmin"

# Data Lake Paths
BRONZE_PATH = f"s3://{BUCKET_NAME}/bronze/"
STAGING_PATH = f"s3://{BUCKET_NAME}/staging/"
INTER_PATH = f"s3://{BUCKET_NAME}/intermediate/"
SILVER_PATH = f"s3://{BUCKET_NAME}/silver/"
MART_PATH = f"s3://{BUCKET_NAME}/mart/"

# Staging Layer Tables
STAGING_TABLES = {
    "sales": "stg_sales",
    "weather": "stg_weather",
    "key": "stg_key",
    "holidays": "stg_holidays",
    "blackfriday": "stg_blackfriday"
}

# Intermediate/Gold Layer Tables
GOLD_TABLES = {
    "sales_features": "sales_features",
    "weather_features": "weather_features"
}

# Feature Engineering Parameters
LAG_PERIODS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]
EWMA_ALPHAS = [0.5, 0.75]

# Weather Codes for One-Hot Encoding
WEATHER_CODES = [
    'RA',  # Rain
    'SN',  # Snow
    'FG',  # Fog
    'BR',  # Mist
    'UP',  # Unknown Precipitation
    'TS',  # Thunderstorm
    'HZ',  # Haze
    'DZ',  # Drizzle
    'SQ',  # Squall
    'FZ',  # Freezing
    'MI',  # Shallow
    'PR',  # Partial
    'BC',  # Patches
    'BL',  # Blowing
    'VC'   # Vicinity
]

# Numeric Weather Columns for Imputation
WEATHER_NUMERIC_COLS = [
    "tmax", "tmin", "tavg", "depart", "dewpoint", "wetbulb",
    "heat", "cool", "sunrise", "sunset", "snowfall", "precip_total",
    "stn_pressure", "sea_level", "result_speed", "result_dir", "avg_speed"
]

# Spark Configuration
SPARK_CONFIGS = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.pyspark.python": "/usr/bin/python3.12",
    "spark.pyspark.driver.python": "python3",
}
