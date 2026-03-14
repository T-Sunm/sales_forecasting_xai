"""
Join Features Pipeline - Mart Layer

Input:  s3a://datalake/intermediate/ (int_sales_with_lags, weather_features)
Output: s3a://datalake/mart/sales_forecast
"""

from pyspark.sql import SparkSession
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "configs"))
from config import STAGING_PATH, INTER_PATH, MART_PATH, SPARK_CONFIGS


def main():
    builder = SparkSession.builder.appName("join-features-mart")
    for key, val in SPARK_CONFIGS.items():
        builder = builder.config(key, val)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    sales = spark.read.parquet(INTER_PATH + "int_sales_with_lags")
    weather = spark.read.parquet(INTER_PATH + "weather_features")
    stg_key = spark.read.parquet(STAGING_PATH + "stg_key")

    joined = (sales
        .join(stg_key, "store_id", "left")
        .join(weather, ["station_id", "date"], "left")
    )

    joined.write.mode("overwrite").parquet(MART_PATH + "sales_forecast")
    spark.stop()


if __name__ == "__main__":
    main()
