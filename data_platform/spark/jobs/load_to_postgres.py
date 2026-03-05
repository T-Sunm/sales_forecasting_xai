import os
import sys
import logging
from pathlib import Path
from pyspark.sql import SparkSession

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# import config
sys.path.append(str(Path(__file__).parent.parent / "configs"))
try:
    from config import STAGING_PATH, INTER_PATH, MART_PATH, SPARK_CONFIGS
except ImportError:
    sys.path.append("/opt/spark/configs")
    try:
        from config import STAGING_PATH, INTER_PATH, MART_PATH, SPARK_CONFIGS
    except ImportError:
        STAGING_PATH = "s3://datalake/staging/"
        INTER_PATH = "s3://datalake/intermediate/"
        MART_PATH = "s3://datalake/mart/"
        SPARK_CONFIGS = {}

PG_HOST = os.environ.get("PG_HOST", "postgres")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DB = os.environ.get("PG_DB", "sales_forecasting")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "changeme")

JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_PROPS = {"user": PG_USER, "password": PG_PASS, "driver": "org.postgresql.Driver"}

TABLES = [
    (STAGING_PATH + "stg_sales", "raw.stg_sales"),
    (STAGING_PATH + "stg_weather", "raw.stg_weather"),
    (STAGING_PATH + "stg_key", "raw.stg_key"),
    (STAGING_PATH + "stg_holidays", "raw.stg_holidays"),
    (STAGING_PATH + "stg_blackfriday", "raw.stg_blackfriday"),
    (INTER_PATH + "int_date_features", "intermediate.int_date_features"),
    (INTER_PATH + "int_store_item_aggregates", "intermediate.int_store_item_aggregates"),
    (INTER_PATH + "int_sales_with_lags", "intermediate.int_sales_with_lags"),
    (INTER_PATH + "int_sales_with_rolling", "intermediate.int_sales_with_rolling"),
    (INTER_PATH + "int_sales_with_ewma", "intermediate.int_sales_with_ewma"),
    (INTER_PATH + "int_active_sales", "intermediate.int_active_sales"),
    (INTER_PATH + "weather_features", "intermediate.weather_features"),
    (MART_PATH + "sales_forecast", "marts.sales_forecast"),
]

def main():
    builder = SparkSession.builder.appName("load-minio-to-postgres")
    for key, val in SPARK_CONFIGS.items():
        builder = builder.config(key, val)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")  # giảm log Spark
    logger.info("Starting migration. db=%s host=%s", PG_DB, PG_HOST)

    for src, dst in TABLES:
        logger.info("Load start. src=%s dst=%s", src, dst)
        try:
            df = spark.read.parquet(src)

            (df.write
               .format("jdbc")
               .option("url", JDBC_URL)
               .option("dbtable", dst)
               .options(**JDBC_PROPS)
               .mode("overwrite")
               .save())

            logger.info("Load ok. dst=%s rows=%d", dst, df.count())
        except Exception:
            logger.exception("Load failed. src=%s dst=%s", src, dst)

    logger.info("Done.")
    spark.stop()

if __name__ == "__main__":
    main()
