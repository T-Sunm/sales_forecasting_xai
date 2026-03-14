import os
from pathlib import Path

import pandas as pd
import yaml
from trino.dbapi import connect
from dotenv import load_dotenv

# Project paths (absolute, calculated from script location)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # ml/scripts -> ml -> project_root
SHARED_DIR = PROJECT_ROOT / "shared"

load_dotenv(PROJECT_ROOT / ".env")

# Trino Configuration (Match backend/src/config.py)
TRINO_HOST = os.getenv("TRINO_HOST", "localhost")
TRINO_PORT = int(os.getenv("TRINO_PORT", "8085"))
TRINO_USER = os.getenv("TRINO_USER", "admin")
TRINO_CATALOG = os.getenv("TRINO_CATALOG", "iceberg")
TRINO_SCHEMA = os.getenv("TRINO_SCHEMA", "analytics")

OUTPUT_DIR = SHARED_DIR / "data" / "processed"
KAGGLE_TEST_CSV = SHARED_DIR / "data" / "data_raw" / "test.csv"
PARAMS_FILE = SHARED_DIR / "params.yaml"


def load_cutoff_date():
    if not PARAMS_FILE.exists():
        return "2014-08-01"
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params.get("prepare", {}).get("cutoff_date", "2014-08-01")


def get_trino_connection():
    return connect(
        host=TRINO_HOST,
        port=TRINO_PORT,
        user=TRINO_USER,
        catalog=TRINO_CATALOG,
        schema=TRINO_SCHEMA,
    )


def prepare_data():
    cutoff_date = load_cutoff_date()

    print(f"Connecting to Trino at {TRINO_HOST}:{TRINO_PORT}...")
    
    # Query logic inspired by backend/src/utils/db_manager.py
    query = """
    SELECT 
        -- Key / Context from Fact Sales
        f.date, f.store_id, f.item_id, f.units, f.log_units,
        -- Lags from Fact Sales
        f.logunits_lag_1, f.logunits_lag_2, f.logunits_lag_3, f.logunits_lag_4, f.logunits_lag_5,
        f.logunits_lag_6, f.logunits_lag_7, f.logunits_lag_14, f.logunits_lag_21, f.logunits_lag_28,
        -- Rolling features from Fact Sales
        f.roll_avg_7d, f.roll_min_7d, f.roll_max_7d, f.roll_std_7d,
        f.roll_avg_14d, f.roll_min_14d, f.roll_max_14d, f.roll_std_14d,
        f.roll_avg_28d, f.roll_min_28d, f.roll_max_28d, f.roll_std_28d,
        -- EWMA and Aggregates from Fact Sales
        f.ewma7_a05, f.ewma7_a075,
        f.store_sum_7d, f.store_mean_7d, f.item_sum_7d, f.item_mean_7d,
        -- Weather Features from Fact Weather
        w.tmax, w.tmin, w.tavg, w.depart, w.dewpoint, w.wetbulb, w.heat, w.cool,
        w.sunrise, w.sunset, w.snowfall, w.preciptotal, w.stnpressure, w.sealevel,
        w.resultspeed, w.resultdir, w.avgspeed,
        w.is_ra, w.is_sn, w.is_fg, w.is_br, w.is_up, w.is_ts, w.is_hz, w.is_dz,
        w.is_sq, w.is_fz, w.is_mi, w.is_pr, w.is_bc, w.is_bl, w.is_vc,
        -- Calendar Features from Dim Date
        d.year, d.month, d.day, d.day_of_week, d.quarter, d.is_weekend, 
        d.is_holiday, d.is_blackfriday,
        d.season_winter, d.season_spring, d.season_summer, d.season_fall
    FROM fact_sales_item_daily f
    JOIN fact_store_weather_daily w ON f.date = w.date AND f.store_id = w.store_id
    JOIN dim_date d ON f.date = d.date
    """

    try:
        with get_trino_connection() as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=columns)
    except Exception as e:
        print(f"Error querying Trino: {e}")
        return

    df["date"] = pd.to_datetime(df["date"])

    # Ensure numeric types (fetched from Trino as objects/strings sometimes)
    for col in df.columns:
        if col not in ["date", "weather_profile_key"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    print(f"Loading Kaggle test IDs from {KAGGLE_TEST_CSV}...")
    if KAGGLE_TEST_CSV.exists():
        df_kaggle_keys = pd.read_csv(KAGGLE_TEST_CSV)
        df_kaggle_keys["date"] = pd.to_datetime(df_kaggle_keys["date"])
        df_kaggle_keys = df_kaggle_keys.rename(columns={"store_nbr": "store_id", "item_nbr": "item_id"})
        df_kaggle_keys["is_kaggle_test"] = 1

        print("Mapping Kaggle test flags...")
        df = df.merge(
            df_kaggle_keys[["date", "store_id", "item_id", "is_kaggle_test"]],
            on=["date", "store_id", "item_id"],
            how="left",
        )
        df["is_kaggle_test"] = df["is_kaggle_test"].fillna(0)
    else:
        print(f"Warning: {KAGGLE_TEST_CSV} not found. Kaggle test flag will be set to 0.")
        df["is_kaggle_test"] = 0

    print(f"Splitting data with cutoff: {cutoff_date}...")
    cutoff = pd.Timestamp(cutoff_date)

    df_test = df[df["is_kaggle_test"] == 1].copy()
    df_pool = df[df["is_kaggle_test"] == 0].copy()

    df_train = df_pool[df_pool["date"] < cutoff].copy()
    df_valid = df_pool[df_pool["date"] >= cutoff].copy()

    print(f"Final Splits: Train({len(df_train)}) | Valid({len(df_valid)}) | KaggleTest({len(df_test)})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving parquet files to {OUTPUT_DIR}...")
    df_train.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    df_valid.to_parquet(OUTPUT_DIR / "valid.parquet", index=False)
    df_test.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()
