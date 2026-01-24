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

def main():
    spark = SparkSession.builder.appName("walmart-intermediate").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"🚀 Starting Sales Features Pipeline")


    # ---------- Load ----------
    # Lưu ý: job staging tạo stg_sales trong folder staging/
    sales = spark.read.parquet(STAGING_PATH + "stg_sales") 
    
    # Ensure types (phòng hờ, dù stg đã cast rồi)
    sales = (sales
             .withColumn("date", F.to_date("date"))
             .withColumn("units", F.col("units").cast("int")))

    w_si = Window.partitionBy("store_id", "item_id").orderBy("date")

    # ---------- 1. int_active_sales ----------
    
    # total_lifetime_units > 0
    sales = sales.withColumn("total_lifetime_units", F.sum("units").over(Window.partitionBy("store_id","item_id")))
    active = sales.filter(F.col("total_lifetime_units") > 0)

    # target scaling: log_units = ln(units + 1)
    active = active.withColumn("log_units", F.log1p(F.col("units").cast("double")))
    
    # Cache active để dùng lại cho các bước sau
    active.cache()
    
    active.write.mode("overwrite").parquet(INTER_PATH + "int_active_sales")
    print("✅ Step 1: int_active_sales saved")

    # ---------- 2. int_sales_with_lags ----------
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    df_lags = active
    for k in lags:
        df_lags = df_lags.withColumn(f"logunits_lag_{k}", F.lag("log_units", k).over(w_si))
    
    # Lưu riêng bảng lags (hoặc có thể gộp nếu muốn như dbt)
    df_lags.write.mode("overwrite").parquet(INTER_PATH + "int_sales_with_lags")
    print("✅ Step 2: int_sales_with_lags saved")

    # ---------- 3. int_sales_with_rolling ----------
    base = "logunits_lag_1"
    
    # Cần join với bảng lags hoặc tính lại lag_1. Ở đây dùng df_lags đã có lag_1
    df_rolling = df_lags

    def add_roll_stats(d, win, prefix):
        w = w_si.rowsBetween(-win+1, 0) # window size N includes current row (which is lag_1) + N-1 previous
        return (d
            .withColumn(f"{prefix}_avg_{win}d", F.avg(F.col(base)).over(w))
            .withColumn(f"{prefix}_min_{win}d", F.min(F.col(base)).over(w))
            .withColumn(f"{prefix}_max_{win}d", F.max(F.col(base)).over(w))
            .withColumn(f"{prefix}_std_{win}d", F.stddev(F.col(base)).over(w))
        )

    for win in [7, 14, 28]:
        df_rolling = add_roll_stats(df_rolling, win, "roll")

    df_rolling.write.mode("overwrite").parquet(INTER_PATH + "int_sales_with_rolling")
    print("✅ Step 3: int_sales_with_rolling saved")

    # ---------- 4. int_sales_with_ewma ----------
    # Đưa vào pandas UDF đúng các cột cần thiết để tối ưu hóa hiệu năng
    base = "logunits_lag_1"
    df_in = df_lags.select("store_id", "item_id", "date", base)
    
    # Định nghĩa schema bằng cách kế thừa type từ input để tránh lỗi mismatch (int/string)
    ewma_schema = StructType(df_in.select("store_id", "item_id", "date").schema.fields)
    ewma_cols = [f"ewma7_a{str(a).replace('.', '')}" for a in EWMA_ALPHAS]
    for c in ewma_cols:
        ewma_schema = ewma_schema.add(c, DoubleType())

    def ewma_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("date")
        s = pdf[base].astype("float64")
        for alpha in EWMA_ALPHAS:
            alpha_str = str(alpha).replace(".", "")
            pdf[f"ewma7_a{alpha_str}"] = s.ewm(alpha=alpha, adjust=False).mean()
            
        return pdf[["store_id", "item_id", "date"] + ewma_cols]

    # applyInPandas grouped by store_item
    df_ewma = df_in.groupBy("store_id", "item_id").applyInPandas(ewma_pdf, schema=ewma_schema)
    df_ewma.write.mode("overwrite").parquet(INTER_PATH + "int_sales_with_ewma")
    print("✅ Step 4: int_sales_with_ewma saved")

    # ---------- 5. int_store_item_aggregates ----------
    # Mang EWMA xuống pipeline bằng cách join với df_rolling
    input_agg = df_rolling.join(df_ewma, ["store_id", "item_id", "date"], "left")
    
    store_daily = (input_agg.groupBy("store_id","date")
                     .agg(F.sum(base).alias("store_sum_day"),
                          F.avg(base).alias("store_mean_day")))
    
    w_store = Window.partitionBy("store_id").orderBy("date").rowsBetween(-6, 0)
    store_ctx = (store_daily
        .withColumn("store_sum_7d",  F.sum("store_sum_day").over(w_store))
        .withColumn("store_mean_7d", F.avg("store_mean_day").over(w_store))
        .select("store_id","date","store_sum_7d","store_mean_7d"))

    item_daily = (input_agg.groupBy("item_id","date")
                    .agg(F.sum(base).alias("item_sum_day"),
                         F.avg(base).alias("item_mean_day")))
    
    w_item = Window.partitionBy("item_id").orderBy("date").rowsBetween(-6, 0)
    item_ctx = (item_daily
        .withColumn("item_sum_7d",  F.sum("item_sum_day").over(w_item))
        .withColumn("item_mean_7d", F.avg("item_mean_day").over(w_item))
        .select("item_id","date","item_sum_7d","item_mean_7d"))

    df_agg = (input_agg
        .join(store_ctx, ["store_id","date"], "left")
        .join(item_ctx,  ["item_id","date"],  "left"))

    df_agg.write.mode("overwrite").parquet(INTER_PATH + "int_store_item_aggregates")
    print("✅ Step 5: int_store_item_aggregates saved")

    # ---------- 6. int_date_features ----------
    # Load holidays/blackfriday từ STAGING (do job staging_transform.py ghi vào staging/)
    hol = spark.read.parquet(STAGING_PATH + "stg_holidays").select(F.to_date("date").alias("date")).withColumn("is_holiday", F.lit(1)).distinct()
    bf  = spark.read.parquet(STAGING_PATH + "stg_blackfriday").select(F.to_date("date").alias("date")).withColumn("is_blackfriday", F.lit(1)).distinct()

    # Join vào dataframe cuối cùng (df_agg tức là đã có đủ rolling + store ctx + lags)
    df_final = (df_agg
        .join(hol, "date", "left")
        .join(bf,  "date", "left")
        .fillna({"is_holiday": 0, "is_blackfriday": 0})
        .withColumn("year",  F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("day",   F.dayofmonth("date"))
        .withColumn("day_of_week", F.dayofweek("date"))
        .withColumn("quarter", F.quarter("date"))
        .withColumn("is_weekend", F.when(F.dayofweek("date").isin([1,7]), 1).otherwise(0))
    )

    df_final = (df_final
        .withColumn("season_winter", F.when(F.col("month").isin([12,1,2]), 1).otherwise(0))
        .withColumn("season_spring", F.when(F.col("month").isin([3,4,5]), 1).otherwise(0))
        .withColumn("season_summer", F.when(F.col("month").isin([6,7,8]), 1).otherwise(0))
        .withColumn("season_fall",   F.when(F.col("month").isin([9,10,11]), 1).otherwise(0))
    )

    df_final.write.mode("overwrite").parquet(INTER_PATH + "int_date_features")
    print("✅ Step 6: int_date_features (FINAL) saved")
    
    active.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()
