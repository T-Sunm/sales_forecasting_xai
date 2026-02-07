import os

import pandas as pd
import yaml
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv("../.env")

PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "changeme")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = "sales_forecasting"

OUTPUT_DIR = "../shared/data/processed"
KAGGLE_TEST_CSV = "../shared/data/data_raw/test.csv"
PARAMS_FILE = "../shared/params.yaml"


def load_cutoff_date():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params.get("prepare", {}).get("cutoff_date", "2014-08-01")


def prepare_data():
    cutoff_date = load_cutoff_date()

    conn_str = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    engine = create_engine(conn_str)

    print(f"Loading Kaggle test IDs from {KAGGLE_TEST_CSV}...")
    df_kaggle_keys = pd.read_csv(KAGGLE_TEST_CSV)
    df_kaggle_keys["date"] = pd.to_datetime(df_kaggle_keys["date"])
    df_kaggle_keys = df_kaggle_keys.rename(columns={"store_nbr": "store_id", "item_nbr": "item_id"})
    df_kaggle_keys["is_kaggle_test"] = 1

    print("Extracting features from DB (joining with dim_date)...")
    query = """
    SELECT 
        sf.*,
        dd.year, dd.month, dd.day, dd.day_of_week, dd.is_weekend, dd.is_holiday, dd.is_blackfriday,
        dd.season_winter, dd.season_spring, dd.season_summer, dd.season_fall
    FROM marts.sales_forecast sf
    JOIN marts.dim_date dd ON sf.date = dd.date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])

    print("Cleaning numeric features...")
    cols_to_fix = [
        "tmax", "tmin", "tavg", "depart", "dewpoint", "wetbulb",
        "heat", "cool", "sunrise", "sunset", "snowfall", "preciptotal",
        "stnpressure", "sealevel", "resultspeed", "resultdir", "avgspeed",
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    print("Mapping Kaggle test flags...")
    df = df.merge(
        df_kaggle_keys[["date", "store_id", "item_id", "is_kaggle_test"]],
        on=["date", "store_id", "item_id"],
        how="left",
    )
    df["is_kaggle_test"] = df["is_kaggle_test"].fillna(0)

    print(f"Splitting data with cutoff: {cutoff_date}...")
    cutoff = pd.Timestamp(cutoff_date)

    df_test = df[df["is_kaggle_test"] == 1].copy()
    df_pool = df[df["is_kaggle_test"] == 0].copy()

    df_train = df_pool[df_pool["date"] < cutoff].copy()
    df_valid = df_pool[df_pool["date"] >= cutoff].copy()

    print(f"Final Splits: Train({len(df_train)}) | Valid({len(df_valid)}) | KaggleTest({len(df_test)})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving parquet files to {OUTPUT_DIR}...")
    df_train.to_parquet(f"{OUTPUT_DIR}/train.parquet", index=False)
    df_valid.to_parquet(f"{OUTPUT_DIR}/valid.parquet", index=False)
    df_test.to_parquet(f"{OUTPUT_DIR}/test.parquet", index=False)

    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()

