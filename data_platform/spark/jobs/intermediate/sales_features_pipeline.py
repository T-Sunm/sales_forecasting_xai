"""
Sales Features Pipeline - Intermediate Layer

Input: s3a://datalake/staging/ (stg_sales, stg_holidays, stg_blackfriday)
Output: s3a://datalake/intermediate/ (int_active_sales, int_sales_with_lags, etc.)
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructType
import pandas as pd

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "configs"))
from config import STAGING_PATH, INTER_PATH, EWMA_ALPHAS

KEYS = ["store_id", "item_id", "date"]

def write_narrow(df, out_path, cols):
    df.select(*cols).write.mode("overwrite").parquet(out_path)

def main():
    spark = SparkSession.builder.appName("walmart-intermediate").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"🚀 Starting Sales Features Pipeline")

    # ---------- Load ----------
    sales = spark.read.parquet(STAGING_PATH + "stg_sales") 
    
    sales = (sales
             .withColumn("date", F.to_date("date"))
             .withColumn("units", F.col("units").cast("int")))

    w_si = Window.partitionBy("store_id", "item_id").orderBy("date")

    # ---------- 1. int_active_sales ----------
    sales = sales.withColumn("total_lifetime_units", F.sum("units").over(Window.partitionBy("store_id","item_id")))
    active = (sales
        .filter(F.col("total_lifetime_units") > 0)
        .withColumn("log_units", F.log1p(F.col("units").cast("double")))
        .drop("total_lifetime_units")
    )
    active.cache()
    
    write_narrow(active, INTER_PATH + "int_active_sales", 
                 ["date","store_id","item_id","units","log_units"])
    print("✅ Step 1: int_active_sales saved")

    # ---------- 2. int_sales_with_lags ----------
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    df_lags = active.select(*KEYS, "log_units")
    for k in lags:
        df_lags = df_lags.withColumn(f"logunits_lag_{k}", F.lag("log_units", k).over(w_si))
    
    lag_cols = KEYS + ["log_units"] + [f"logunits_lag_{k}" for k in lags]
    write_narrow(df_lags, INTER_PATH + "int_sales_with_lags", lag_cols)
    print("✅ Step 2: int_sales_with_lags saved")

    # ---------- 3. int_sales_with_rolling ----------
    base_col = "logunits_lag_1"
    df_roll_in = df_lags.select(*KEYS, base_col)

    def add_roll_stats(d, win):
        w = w_si.rowsBetween(-win+1, 0)
        return (d
            .withColumn(f"roll_avg_{win}d", F.avg(F.col(base_col)).over(w))
            .withColumn(f"roll_min_{win}d", F.min(F.col(base_col)).over(w))
            .withColumn(f"roll_max_{win}d", F.max(F.col(base_col)).over(w))
            .withColumn(f"roll_std_{win}d", F.stddev(F.col(base_col)).over(w))
        )

    df_rolling = df_roll_in
    for win in [7, 14, 28]:
        df_rolling = add_roll_stats(df_rolling, win)

    rolling_cols = KEYS + [c for c in df_rolling.columns if c.startswith("roll_")]
    write_narrow(df_rolling, INTER_PATH + "int_sales_with_rolling", rolling_cols)
    print("✅ Step 3: int_sales_with_rolling saved")

    # ---------- 4. int_sales_with_ewma ----------
    df_in = df_lags.select("store_id", "item_id", "date", base_col)
    
    ewma_schema = StructType(df_in.select("store_id", "item_id", "date").schema.fields)
    ewma_cols_names = [f"ewma7_a{str(a).replace('.', '')}" for a in EWMA_ALPHAS]
    for c in ewma_cols_names:
        ewma_schema = ewma_schema.add(c, DoubleType())

    def ewma_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("date")
        s = pdf[base_col].astype("float64")
        for alpha in EWMA_ALPHAS:
            alpha_str = str(alpha).replace(".", "")
            pdf[f"ewma7_a{alpha_str}"] = s.ewm(alpha=alpha, adjust=False).mean()
        return pdf[["store_id", "item_id", "date"] + ewma_cols_names]

    df_ewma = df_in.groupBy("store_id", "item_id").applyInPandas(ewma_pdf, schema=ewma_schema)
    
    write_narrow(df_ewma, INTER_PATH + "int_sales_with_ewma", KEYS + ewma_cols_names)
    print("✅ Step 4: int_sales_with_ewma saved")

    # ---------- 5. int_store_item_aggregates ----------
    # Input narrow for aggregates
    input_agg = (df_roll_in
        .join(df_ewma, KEYS, "left")
    )
    
    store_daily = (input_agg.groupBy("store_id","date")
                     .agg(F.sum(base_col).alias("store_sum_day"),
                          F.avg(base_col).alias("store_mean_day")))
    
    w_store = Window.partitionBy("store_id").orderBy("date").rowsBetween(-6, 0)
    store_ctx = (store_daily
        .withColumn("store_sum_7d",  F.sum("store_sum_day").over(w_store))
        .withColumn("store_mean_7d", F.avg("store_mean_day").over(w_store))
        .select("store_id","date","store_sum_7d","store_mean_7d"))
    
    item_daily = (input_agg.groupBy("item_id","date")
                    .agg(F.sum(base_col).alias("item_sum_day"),
                         F.avg(base_col).alias("item_mean_day")))
    
    w_item = Window.partitionBy("item_id").orderBy("date").rowsBetween(-6, 0)
    item_ctx = (item_daily
        .withColumn("item_sum_7d",  F.sum("item_sum_day").over(w_item))
        .withColumn("item_mean_7d", F.avg("item_mean_day").over(w_item))
        .select("item_id","date","item_sum_7d","item_mean_7d"))

    df_ctx = (input_agg.select(*KEYS)
        .join(store_ctx, ["store_id","date"], "left")
        .join(item_ctx,  ["item_id","date"],  "left"))

    ctx_cols = KEYS + ["store_sum_7d","store_mean_7d","item_sum_7d","item_mean_7d"]
    write_narrow(df_ctx, INTER_PATH + "int_store_item_aggregates", ctx_cols)
    print("✅ Step 5: int_store_item_aggregates saved")

    # ---------- 6. int_date_features ----------
    dates = active.select(F.col("date")).distinct()

    hol = spark.read.parquet(STAGING_PATH + "stg_holidays").select(F.to_date("date").alias("date")).withColumn("is_holiday", F.lit(1)).distinct()
    bf  = spark.read.parquet(STAGING_PATH + "stg_blackfriday").select(F.to_date("date").alias("date")).withColumn("is_blackfriday", F.lit(1)).distinct()

    dim_date = (dates
        .join(hol, "date", "left")
        .join(bf,  "date", "left")
        .fillna({"is_holiday": 0, "is_blackfriday": 0})
        .withColumn("year",  F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("day",   F.dayofmonth("date"))
        .withColumn("day_of_week", F.dayofweek("date"))
        .withColumn("quarter", F.quarter("date"))
        .withColumn("is_weekend", F.when(F.dayofweek("date").isin([1,7]), 1).otherwise(0))
        .withColumn("season_winter", F.when(F.col("month").isin([12,1,2]), 1).otherwise(0))
        .withColumn("season_spring", F.when(F.col("month").isin([3,4,5]), 1).otherwise(0))
        .withColumn("season_summer", F.when(F.col("month").isin([6,7,8]), 1).otherwise(0))
        .withColumn("season_fall",   F.when(F.col("month").isin([9,10,11]), 1).otherwise(0))
    )

    date_cols = ["date","year","month","day","day_of_week","quarter","is_weekend",
                 "is_holiday","is_blackfriday",
                 "season_winter","season_spring","season_summer","season_fall"]
    write_narrow(dim_date, INTER_PATH + "int_date_features", date_cols)
    print("✅ Step 6: int_date_features (FINAL) saved")
    
    active.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
