import logging
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2 import OperationalError, DatabaseError

from src.config import PG_DSN

logger = logging.getLogger(__name__)


class DatabaseServiceError(Exception):
    """Raised when a PostgreSQL query fails. Converted to HTTP 503 by the app exception handler."""
    pass


def get_connection():
    return psycopg2.connect(PG_DSN)


def run_query(query: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Run a SQL query against PostgreSQL. Raises DatabaseServiceError on DB failures."""
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()
    except (DatabaseError, OperationalError) as e:
        logger.error("PostgreSQL query failed: %s | query=%s", e, str(query)[:200])
        raise DatabaseServiceError(str(e)) from e


def fetch_ml_features(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch full feature engineering set from PostgreSQL (marts schema).
    Joins Fact Sales, Fact Weather, and Dim Date from dbt marts layer.
    """
    query = """
    SELECT
        f.date, f.store_id, f.item_id, f.units, f.log_units,
        f.logunits_lag_1, f.logunits_lag_2, f.logunits_lag_3, f.logunits_lag_4,
        f.logunits_lag_5, f.logunits_lag_6, f.logunits_lag_7, f.logunits_lag_14,
        f.logunits_lag_21, f.logunits_lag_28,
        f.roll_avg_7d, f.roll_min_7d, f.roll_max_7d, f.roll_std_7d,
        f.roll_avg_14d, f.roll_min_14d, f.roll_max_14d, f.roll_std_14d,
        f.roll_avg_28d, f.roll_min_28d, f.roll_max_28d, f.roll_std_28d,
        f.ewma7_a05, f.ewma7_a075,
        f.store_sum_7d, f.store_mean_7d, f.item_sum_7d, f.item_mean_7d,
        w.tmax, w.tmin, w.tavg, w.depart, w.dewpoint, w.wetbulb, w.heat, w.cool,
        w.sunrise, w.sunset, w.snowfall, w.preciptotal, w.stnpressure, w.sealevel,
        w.resultspeed, w.resultdir, w.avgspeed,
        w.is_ra, w.is_sn, w.is_fg, w.is_br, w.is_up, w.is_ts, w.is_hz, w.is_dz,
        w.is_sq, w.is_fz, w.is_mi, w.is_pr, w.is_bc, w.is_bl, w.is_vc,
        d.year, d.month, d.day, d.day_of_week, d.quarter,
        d.is_weekend, d.is_holiday, d.is_blackfriday,
        d.season_winter, d.season_spring, d.season_summer, d.season_fall
    FROM marts.fact_sales_item_daily f
    JOIN marts.fact_store_weather_daily w ON f.date = w.date AND f.store_id = w.store_id
    JOIN marts.dim_date d ON f.date = d.date
    """
    if limit:
        query += f" LIMIT {limit}"

    df = run_query(query)
    # Coerce all numeric columns to float, except date/store/item ids
    numeric_cols = df.columns.drop(['date', 'store_id', 'item_id'])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df
