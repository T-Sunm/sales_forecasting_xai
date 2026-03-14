"""
Sales EWMA Pipeline - Intermediate Layer

Input : s3a://datalake/staging/parquet/stg_sales
Output: s3a://datalake/intermediate/int_sales_with_ewma

Mô tả : Tính Exponential Weighted Moving Average (EWMA) cho log_units.
        Dùng applyInPandas vì không có hàm EWM tương đương trong Spark SQL.
        Các features khác (lags, rolling, aggregates, date) do dbt đảm nhận.
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructType
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "configs"))
from config import STAGING_PATH, INTER_PATH, EWMA_ALPHAS, SPARK_CONFIGS


def main():
    builder = SparkSession.builder.appName("ewma-sales-features")
    for key, val in SPARK_CONFIGS.items():
        builder = builder.config(key, val)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # --- Load & filter active sales ---
    w_si = Window.partitionBy("store_id", "item_id").orderBy("date")

    sales = (
        spark.read.parquet(STAGING_PATH + "stg_sales")
        .withColumn("date", F.to_date("date"))
        .withColumn("units", F.col("units").cast("int"))
        .withColumn("total_lifetime_units", F.sum("units").over(Window.partitionBy("store_id", "item_id")))
        .filter(F.col("total_lifetime_units") > 0)
        .withColumn("log_units", F.log1p(F.col("units").cast("double")))
        .withColumn("logunits_lag_1", F.lag("log_units", 1).over(w_si))
        .select("store_id", "item_id", "date", "logunits_lag_1")
    )

    # --- EWMA via applyInPandas ---
    ewma_col_names = [f"ewma7_a{str(a).replace('.', '')}" for a in EWMA_ALPHAS]

    schema = StructType(sales.select("store_id", "item_id", "date").schema.fields)
    for col in ewma_col_names:
        schema = schema.add(col, DoubleType())

    def compute_ewma(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("date")
        s = pdf["logunits_lag_1"].astype("float64")
        for alpha in EWMA_ALPHAS:
            pdf[f"ewma7_a{str(alpha).replace('.', '')}"] = s.ewm(alpha=alpha, adjust=False).mean()
        return pdf[["store_id", "item_id", "date"] + ewma_col_names]

    df_ewma = sales.groupBy("store_id", "item_id").applyInPandas(compute_ewma, schema=schema)

    df_ewma.write.mode("overwrite").parquet(INTER_PATH + "int_sales_with_ewma")

    spark.stop()


if __name__ == "__main__":
    main()
