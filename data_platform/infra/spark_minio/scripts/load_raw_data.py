import pandas as pd
from sqlalchemy import create_engine, text 
from pathlib import Path

DATA_RAW_DIR = Path("E:/AIO/Project/sales_forecasting_xai/shared/data/data_raw")

engine = create_engine('postgresql://postgres:changeme@localhost:5432/postgres')

with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
    conn.commit()
    print("✅ Schema 'raw' created/verified")

# Mapping: csv filename -> table name
TABLES = {
    "train.csv": "raw_sales",
    "weather.csv": "raw_weather",
    "key.csv": "raw_key",
}

for csv_file, table_name in TABLES.items():
    df = pd.read_csv(DATA_RAW_DIR / csv_file)
    df.to_sql(table_name, engine, schema='raw', if_exists='replace', index=False)
    print(f"Loaded {csv_file} -> raw.{table_name}")