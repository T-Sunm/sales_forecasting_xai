"""
Weather Features Pipeline - Intermediate Layer

This script transforms input weather data with:
- Missing value imputation (station mean → global mean)
- Weather code parsing (one-hot encoding)

Input: Configured input path from staging layer
Output: Configured output path for features
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "configs"))
from config import STAGING_PATH, INTER_PATH, WEATHER_CODES, WEATHER_NUMERIC_COLS, SPARK_CONFIGS

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def create_spark_session():
    builder = SparkSession.builder.appName("weather-features-intermediate")
    for key, val in SPARK_CONFIGS.items():
        builder = builder.config(key, val)
    return builder.getOrCreate()


def load_input_data(spark):
    """Load weather data from configured input path"""
    input_data = spark.read.parquet(STAGING_PATH + "stg_weather")
    return input_data.withColumn("date", F.to_date("date"))


def impute_missing_values(df):
    """
    Impute missing values using hierarchical strategy:
    1. Station mean (join station mean)
    2. Global mean (broadcast cross join global mean)
    """
    numeric_cols = [c for c in WEATHER_NUMERIC_COLS if c in df.columns]
    
    # Calculate station means
    station_means = df.groupBy("station_id").agg(*[
        F.avg(F.col(c)).alias(f"{c}_stn_mean") for c in numeric_cols
    ])
    
    # Calculate global means (will be broadcasted)
    global_means = df.agg(*[
        F.avg(F.col(c)).alias(f"{c}_glv_mean") for c in numeric_cols
    ])
    
    # Hierarchical join and imputation
    df = df.join(station_means, on="station_id", how="left")
    df = df.crossJoin(F.broadcast(global_means))
    
    for col in numeric_cols:
        df = df.withColumn(
            col,
            F.coalesce(F.col(col), F.col(f"{col}_stn_mean"), F.col(f"{col}_glv_mean"))
        )
    
    # Cleanup intermediate columns
    drop_cols = [f"{c}_stn_mean" for c in numeric_cols] + [f"{c}_glv_mean" for c in numeric_cols]
    return df.drop(*drop_cols)


def parse_weather_codes(df):
    """
    Parse codesum column into one-hot encoded features.
    
    Weather codes based on dbt macro:
    RA, SN, FG, BR, UP, TS, HZ, DZ, SQ, FZ, MI, PR, BC, BL, VC
    """
    weather_codes = WEATHER_CODES
    
    df = df.withColumn("codesum_clean", F.trim(F.coalesce("codesum", F.lit(""))))
    
    for code in weather_codes:
        col_name = f"is_{code.lower()}"
        df = df.withColumn(
            col_name,
            F.when(F.col("codesum_clean").contains(code), 1).otherwise(0)
        )
    
    return df.drop("codesum_clean")


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    df = load_input_data(spark)
    df = impute_missing_values(df)
    df = parse_weather_codes(df)
    df.write.mode("overwrite").parquet(INTER_PATH + "weather_features")
    spark.stop()


if __name__ == "__main__":
    main()
