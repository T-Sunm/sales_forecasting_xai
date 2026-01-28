from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "configs"))
from config import STAGING_PATH, SPARK_CONFIGS

IN_BASE = STAGING_PATH + "parquet/"
OUT_BASE = STAGING_PATH

builder = SparkSession.builder.appName("walmart-staging-transform")
for key, val in SPARK_CONFIGS.items():
    builder = builder.config(key, val)

spark = builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def clean_weather_numeric(col_name, trace_value=0.0):
    """Clean weather numeric columns: M->null, T->trace_value, empty->null"""
    x = F.trim(F.col(col_name).cast("string"))
    x = F.when(x.isNull() | (x == "") | (x == "M"), F.lit(None)) \
         .when(x == "T", F.lit(trace_value)) \
         .otherwise(x)
    x = F.regexp_replace(x, r"[^0-9\.\-]+", "")
    return F.when(F.trim(x) == "", F.lit(None)).otherwise(x.cast("double"))


def transform_key():
    raw = spark.read.parquet(IN_BASE + "key")
    stg = raw.select(
        F.col("store_nbr").alias("store_id"),
        F.col("station_nbr").alias("station_id"),
    )
    stg.write.mode("overwrite").parquet(OUT_BASE + "stg_key")


def transform_sales():
    raw = spark.read.parquet(IN_BASE + "train")
    stg = (
        raw
        .withColumn("date", F.to_date(F.col("date")))
        .withColumn("units", F.col("units").cast("int"))
        .withColumnRenamed("store_nbr", "store_id")
        .withColumnRenamed("item_nbr", "item_id")
    )
    stg.write.mode("overwrite").parquet(OUT_BASE + "stg_sales")


def transform_weather():
    raw = spark.read.parquet(IN_BASE + "weather")
    
    numeric_cols = [
        "tmax", "tmin", "tavg",
        "dewpoint", "wetbulb",
        "preciptotal", "snowfall",
        "resultspeed", "resultdir", "avgspeed",
    ]
    
    stg = raw \
        .withColumn("date", F.to_date(F.col("date"))) \
        .withColumnRenamed("station_nbr", "station_id")
    
    for col in numeric_cols:
        if col in stg.columns:
            stg = stg.withColumn(col, clean_weather_numeric(col))
    
    stg.write.mode("overwrite").parquet(OUT_BASE + "stg_weather")


def transform_holidays():
    raw = spark.read.parquet(IN_BASE + "holidays")
    stg = raw.select(F.col("date"))
    stg.write.mode("overwrite").parquet(OUT_BASE + "stg_holidays")


def transform_blackfriday():
    raw = spark.read.parquet(IN_BASE + "blackfriday")
    stg = raw.select(F.col("date"))
    stg.write.mode("overwrite").parquet(OUT_BASE + "stg_blackfriday")


if __name__ == "__main__":
    transform_key()
    transform_sales()
    transform_weather()
    transform_holidays()
    transform_blackfriday()
    spark.stop()
