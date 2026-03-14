# === OLD CODE (inline config) replace by spark-defaults.conf ===
# from pyspark.sql import SparkSession
#
# MINIO_ENDPOINT = "http://minio:9000"
# BUCKET = "datalake"
#
# spark = (
#     SparkSession.builder
#     .appName("walmart-staging-ingest")
#     .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
#     .config("spark.hadoop.fs.s3a.path.style.access", "true")
#     .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
#     .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
#     .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
#     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
#     .config("spark.hadoop.fs.s3a.aws.credentials.provider",
#             "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
#     .getOrCreate()
# )
#
# RAW_BASE = f"s3a://{BUCKET}/staging/raw/"
# PARQUET_BASE = f"s3a://{BUCKET}/staging/parquet/"
#
# CSV_FILES = ["train.csv", "key.csv", "weather.csv", "test.csv", "holidays.csv", "blackfriday.csv"]
#
#
# def ingest_csv(name: str):
#     df = (spark.read
#           .option("header", "true")
#           .option("inferSchema", "true")
#           .csv(RAW_BASE + name))
#     out_path = PARQUET_BASE + name.replace(".csv", "")
#     df.write.mode("overwrite").parquet(out_path)
#     print(f"✅ {name} -> {out_path}")
#
#
# for f in CSV_FILES:
#     try:
#         ingest_csv(f)
#     except Exception as e:
#         print(f"⚠️ Skip {f}: {e}")
#
# spark.stop()
# === END OLD CODE ===


from pyspark.sql import SparkSession

BUCKET = "datalake"
RAW_BASE = f"s3://{BUCKET}/staging/raw/"
PARQUET_BASE = f"s3://{BUCKET}/staging/parquet/"
CSV_FILES = ["train.csv", "key.csv", "weather.csv", "test.csv", "holidays.csv", "blackfriday.csv"]

builder = SparkSession.builder.appName("walmart-staging-ingest")
spark = builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def ingest_csv(name: str):
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(RAW_BASE + name)
    out_path = PARQUET_BASE + name.replace(".csv", "")
    df.write.mode("overwrite").parquet(out_path)
    
    # Save as Iceberg table for standard Lakehouse architecture
    table_name = "default.raw_" + name.replace(".csv", "")
    # Fix spaces and special characters in column names to be safely used in Iceberg
    df_iceberg = df
    for col_name in df_iceberg.columns:
        df_iceberg = df_iceberg.withColumnRenamed(col_name, col_name.strip().replace(' ', '_'))
        
    df_iceberg.write.format("iceberg").mode("overwrite").saveAsTable(table_name)
    print(f"✅ {name} -> {out_path} & {table_name}")


for f in CSV_FILES:
    try:
        ingest_csv(f)
    except Exception as e:
        print(f"⚠️ Skip {f}: {e}")

spark.stop()
