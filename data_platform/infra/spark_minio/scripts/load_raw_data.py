# import pandas as pd
# from sqlalchemy import create_engine, text 
# from pathlib import Path

# DATA_RAW_DIR = Path("E:/AIO/Project/sales_forecasting_xai/shared/data/data_raw")

# engine = create_engine('postgresql://postgres:changeme@localhost:5432/postgres')

# with engine.connect() as conn:
#     conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
#     conn.commit()
#     print("✅ Schema 'raw' created/verified")

# # Mapping: csv filename -> table name
# TABLES = {
#     "train.csv": "raw_sales",
#     "weather.csv": "raw_weather",
#     "key.csv": "raw_key",
# }

# for csv_file, table_name in TABLES.items():
#     df = pd.read_csv(DATA_RAW_DIR / csv_file)
#     df.to_sql(table_name, engine, schema='raw', if_exists='replace', index=False)
#     print(f"Loaded {csv_file} -> raw.{table_name}")


import os
from pathlib import Path
from minio import Minio
from minio.error import S3Error

ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")  # host: localhost:9000; container: minio:9000
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

BUCKET = os.getenv("MINIO_BUCKET", "datalake")
LOCAL_DIR = Path(
    os.getenv(
        "LOCAL_DIR",
        "E:/AIO/Project/sales_forecasting_xai/shared/data/data_raw"
    )
)
PREFIX = os.getenv("PREFIX", "staging/raw/")  # trong bucket

def main():
    client = Minio(ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=SECURE)

    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    for csv_file in ["train.csv", "test.csv", "weather.csv", "key.csv"]:
        file_path = LOCAL_DIR / csv_file
        if not file_path.exists():
            print(f"Warning: {csv_file} not found in {LOCAL_DIR}, skipping...")
            continue
        
        object_name = f"{PREFIX}{csv_file}"
        client.fput_object(BUCKET, object_name, str(file_path), content_type="text/csv")
        print(f"Uploaded {csv_file} -> s3://{BUCKET}/{object_name}")

if __name__ == "__main__":
    try:
        main()
    except S3Error as e:
        raise SystemExit(e)
