import logging

import pandas as pd
from trino.dbapi import connect
from trino.exceptions import DatabaseError, OperationalError

from src.config import TRINO_HOST, TRINO_PORT, TRINO_USER, TRINO_CATALOG, TRINO_SCHEMA

logger = logging.getLogger(__name__)


class TrinoServiceError(Exception):
    """Raised when a Trino query fails. Converted to HTTP 503 by the app exception handler."""
    pass


def get_connection():
    return connect(
        host=TRINO_HOST,
        port=TRINO_PORT,
        user=TRINO_USER,
        catalog=TRINO_CATALOG,
        schema=TRINO_SCHEMA,
    )


def run_query(query: str, params: tuple = None) -> pd.DataFrame:
    """Run a SQL query against Trino. Raises TrinoServiceError on DB failures."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, list(params) if params else [])
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return pd.DataFrame(rows, columns=columns)
    except (DatabaseError, OperationalError) as e:
        logger.error("Trino query failed: %s | query=%s", e, query[:200])
        raise TrinoServiceError(str(e)) from e


def fetch_ml_features(limit: int = None) -> pd.DataFrame:
    """
    Fetch full feature engineering set (55 features) from Trino by joining
    Fact Sales, Fact Weather, and Dim Date.
    """
    query = """
    SELECT 
        -- Key / Context
        f.date, f.store_id, f.item_id, f.units, f.log_units,
        -- Lags
        f.logunits_lag_1, f.logunits_lag_2, f.logunits_lag_3, f.logunits_lag_4, f.logunits_lag_5,
        f.logunits_lag_6, f.logunits_lag_7, f.logunits_lag_14, f.logunits_lag_21, f.logunits_lag_28,
        f.roll_avg_7d, f.roll_min_7d, f.roll_max_7d, f.roll_std_7d,
        f.roll_avg_14d, f.roll_min_14d, f.roll_max_14d, f.roll_std_14d,
        f.roll_avg_28d, f.roll_min_28d, f.roll_max_28d, f.roll_std_28d,
        f.ewma7_a05, f.ewma7_a075,
        f.store_sum_7d, f.store_mean_7d, f.item_sum_7d, f.item_mean_7d,
        -- Weather Numeric
        w.tmax, w.tmin, w.tavg, w.depart, w.dewpoint, w.wetbulb, w.heat, w.cool,
        w.sunrise, w.sunset, w.snowfall, w.preciptotal, w.stnpressure, w.sealevel,
        w.resultspeed, w.resultdir, w.avgspeed,
        -- Weather codes (Flags)
        w.is_ra, w.is_sn, w.is_fg, w.is_br, w.is_up, w.is_ts, w.is_hz, w.is_dz,
        w.is_sq, w.is_fz, w.is_mi, w.is_pr, w.is_bc, w.is_bl, w.is_vc,
        -- Calendar
        d.year, d.month, d.day, d.day_of_week, d.quarter, d.is_weekend, d.is_holiday, d.is_blackfriday,
        d.season_winter, d.season_spring, d.season_summer, d.season_fall
    FROM fact_sales_item_daily f
    JOIN fact_store_weather_daily w ON f.date = w.date AND f.store_id = w.store_id
    JOIN dim_date d ON f.date = d.date
    """
    if limit:
        query += f" LIMIT {limit}"
    
    return run_query(query)
