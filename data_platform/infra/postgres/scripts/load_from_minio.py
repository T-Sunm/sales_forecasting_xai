import os
import sys
import psycopg2
from pathlib import Path
from pyspark.sql import SparkSession

# Add configs path to sys.path to import config.py
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "spark" / "configs"))
try:
    from config import STAGING_PATH, INTER_PATH, SPARK_CONFIGS
except ImportError:
    # Fallback if config.py is not reachable
    STAGING_PATH = "s3a://datalake/staging/"
    INTER_PATH   = "s3a://datalake/intermediate/"
    SPARK_CONFIGS = {}

# PostgreSQL Connection from Environment Variables
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DB   = os.environ.get("PG_DB", "sales_forecasting")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "changeme")

# Management Settings
ADMIN_DB = os.environ.get("PG_ADMIN_DB", "postgres")
RESET_DB = os.environ.get("RESET_DB", "0") == "1"

# Spark Connection Settings (for running from host to Docker)
SPARK_MASTER = os.environ.get("SPARK_MASTER", "spark://localhost:7077")
SPARK_DRIVER_HOST = os.environ.get("SPARK_DRIVER_HOST", "host.docker.internal")

JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_PROPS = {
    "user": PG_USER,
    "password": PG_PASS,
    "driver": "org.postgresql.Driver",
}

# Tables to load: (Source MinIO Path, Destination Postgres Table)
TABLES = [
    # 1. Staging Layer -> 'raw' schema in Postgres
    (STAGING_PATH + "stg_sales",       "raw.stg_sales"),
    (STAGING_PATH + "stg_weather",     "raw.stg_weather"),
    (STAGING_PATH + "stg_key",         "raw.stg_key"),
    (STAGING_PATH + "stg_holidays",    "raw.stg_holidays"),
    (STAGING_PATH + "stg_blackfriday", "raw.stg_blackfriday"),

    # 2. Intermediate Layer -> 'intermediate' schema in Postgres
    (INTER_PATH + "int_date_features",         "intermediate.int_date_features"),
    (INTER_PATH + "int_store_item_aggregates", "intermediate.int_store_item_aggregates"),
    (INTER_PATH + "int_sales_with_lags",       "intermediate.int_sales_with_lags"),
    (INTER_PATH + "int_sales_with_rolling",    "intermediate.int_sales_with_rolling"),
    (INTER_PATH + "int_sales_with_ewma",       "intermediate.int_sales_with_ewma"),
    (INTER_PATH + "int_active_sales",          "intermediate.int_active_sales"),
    (INTER_PATH + "weather_features",          "intermediate.weather_features"),
]

def reset_database():
    """Drops and recreates the target database. Requires connection to an admin DB."""
    print(f"🔄 Resetting database: {PG_DB}...")
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=ADMIN_DB,
            user=PG_USER, password=PG_PASS
        )
        conn.autocommit = True
        cur = conn.cursor()

        # Terminate other connections to the target DB
        cur.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s
              AND pid <> pg_backend_pid();
        """, (PG_DB,))

        # Drop and Recreate
        cur.execute(f'DROP DATABASE IF EXISTS "{PG_DB}";')
        cur.execute(f'CREATE DATABASE "{PG_DB}";')
        
        cur.close()
        conn.close()
        print(f"✅ Database {PG_DB} reset successfully.")
    except Exception as e:
        print(f"❌ Error resetting database: {str(e)}")
        raise

def ensure_schemas():
    """Creates necessary schemas in the target database."""
    print(f"🛠 Ensuring schemas exist in {PG_DB}...")
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB,
            user=PG_USER, password=PG_PASS
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        cur.execute("CREATE SCHEMA IF NOT EXISTS raw;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS intermediate;")
        cur.execute("CREATE SCHEMA IF NOT EXISTS marts;")
        
        cur.close()
        conn.close()
        print("✅ Schemas (raw, intermediate, marts) verified.")
    except Exception as e:
        print(f"❌ Error creating schemas: {str(e)}")
        raise

def main():
    # 0. Database Maintenance
    if RESET_DB:
        reset_database()
    ensure_schemas()

    # 1. Spark Processing
    builder = SparkSession.builder \
        .appName("load-minio-to-postgres") \
        .master(SPARK_MASTER) \
        .config("spark.driver.host", SPARK_DRIVER_HOST)

    # Apply S3/MinIO/Tuning configs from config.py
    for key, val in SPARK_CONFIGS.items():
        builder = builder.config(key, val)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"🚀 Starting data migration from MinIO to PostgreSQL ({PG_HOST})")
    print(f"📍 Spark Master: {SPARK_MASTER}")

    for src, dst in TABLES:
        try:
            print(f"📦 Reading parquet from: {src}")
            df = spark.read.parquet(src)
            
            print(f"📤 Writing to JDBC table: {dst}")
            (df.write
               .format("jdbc")
               .option("url", JDBC_URL)
               .option("dbtable", dst)
               .options(**JDBC_PROPS)
               .mode("overwrite")
               .save())
            
            print(f"✅ Loaded {dst} successfully.")
        except Exception as e:
            print(f"❌ Failed to load {src} to {dst}: {str(e)}")

    print("🎉 Load process completed!")
    spark.stop()

if __name__ == "__main__":
    main()
