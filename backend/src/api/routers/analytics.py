from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Any
import pandas as pd
from datetime import date, timedelta

from src.utils.db_manager import run_query

router = APIRouter(prefix="/analytics", tags=["Analytics"])

# ── PostgreSQL table names (marts schema — dbt output layer) ───────────────────
_FACT = "marts.fact_sales_item_daily"          # grain: date × store_id × item_id

# Aggregated views built inline via CTEs
_DATE_SALES = f"""(
    SELECT date,
           SUM(units)  AS total_units,
           COUNT(*)    AS sales_records
    FROM {_FACT}
    GROUP BY date
)"""

_STORE_DAY = f"""(
    SELECT date,
           store_id,
           SUM(units)  AS total_units,
           COUNT(*)    AS sales_records
    FROM {_FACT}
    GROUP BY date, store_id
)"""


@router.get("/filters")
async def get_dashboard_filters():
    """Get initial filter values (date range, stores)"""
    date_bounds = run_query(
        f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {_FACT}"
    )
    if date_bounds.empty:
        raise HTTPException(503, "No data available in fact_sales_item_daily")

    stores_df = run_query(
        f"SELECT DISTINCT store_id FROM {_STORE_DAY} t ORDER BY store_id"
    )
    return {
        "min_date": str(date_bounds["min_date"].iloc[0]),
        "max_date": str(date_bounds["max_date"].iloc[0]),
        "stores": stores_df["store_id"].tolist()
    }


@router.get("/items")
async def get_items(store_id: Optional[int] = None):
    """Get items, optionally filtered by store"""
    if store_id is None:
        items_df = run_query(
            f"SELECT DISTINCT item_id FROM {_FACT} ORDER BY item_id"
        )
    else:
        items_df = run_query(
            f"SELECT DISTINCT item_id FROM {_FACT} WHERE store_id = %s ORDER BY item_id",
            (store_id,)
        )
    return {"items": items_df["item_id"].tolist()}


@router.get("/kpis")
async def get_kpis(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None
):
    """Calculate KPI metrics for the given range"""
    if store_id is None:
        grain_table = _DATE_SALES
        where_clause = "date BETWEEN %s AND %s"
        base_params = (start_date, end_date)
    else:
        grain_table = _STORE_DAY
        where_clause = "store_id = %s AND date BETWEEN %s AND %s"
        base_params = (store_id, start_date, end_date)

    dates = run_query(
        f"SELECT MIN(date) as mn, MAX(date) as mx FROM {grain_table} t WHERE {where_clause}",
        base_params
    )
    if dates.empty or dates.iloc[0, 0] is None:
        return {"total_units": 0, "avg_daily": 0, "growth_pct": 0}

    min_d, max_d = dates.iloc[0, 0], dates.iloc[0, 1]

    if isinstance(min_d, str):
        min_d = date.fromisoformat(min_d)
    if isinstance(max_d, str):
        max_d = date.fromisoformat(max_d)

    mid_d = min_d + timedelta(days=(max_d - min_d).days // 2)

    if store_id is None:
        query = f"""
            SELECT
                SUM(total_units) as total_units,
                AVG(total_units) as avg_daily,
                SUM(sales_records) as total_records,
                SUM(CASE WHEN date <= %s THEN total_units ELSE 0 END) as p1_units,
                SUM(CASE WHEN date >  %s THEN total_units ELSE 0 END) as p2_units
            FROM {grain_table} t
            WHERE date BETWEEN %s AND %s
        """
        kpi_params = (mid_d, mid_d, start_date, end_date)
    else:
        query = f"""
            SELECT
                SUM(total_units) as total_units,
                AVG(total_units) as avg_daily,
                SUM(sales_records) as total_records,
                SUM(CASE WHEN date <= %s THEN total_units ELSE 0 END) as p1_units,
                SUM(CASE WHEN date >  %s THEN total_units ELSE 0 END) as p2_units
            FROM {grain_table} t
            WHERE store_id = %s AND date BETWEEN %s AND %s
        """
        kpi_params = (mid_d, mid_d, store_id, start_date, end_date)

    kpi_df = run_query(query, kpi_params)
    if kpi_df.empty:
        return {"total_units": 0, "avg_daily": 0}

    days_df = run_query(
        f"SELECT COUNT(DISTINCT date) as cnt FROM {grain_table} t WHERE {where_clause}",
        base_params
    )

    row = kpi_df.iloc[0]
    p1 = row["p1_units"] or 0
    p2 = row["p2_units"] or 0
    growth = ((p2 - p1) / p1 * 100) if p1 > 0 else 0

    return {
        "total_units": float(row["total_units"] or 0),
        "avg_daily": float(row["avg_daily"] or 0),
        "total_records": int(row["total_records"] or 0),
        "growth_pct": float(growth),
        "days_count": int(days_df.iloc[0, 0] or 1),
        "p1_units": float(p1),
        "p2_units": float(p2)
    }


@router.get("/trends")
async def get_trends(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None
):
    """Get time-series trends"""
    if store_id is None:
        query = f"""
            SELECT TO_CHAR(date, 'YYYY-MM-DD') as date, SUM(units) as units
            FROM {_FACT}
            WHERE date BETWEEN %s AND %s
            GROUP BY date
            ORDER BY date
        """
        params = (start_date, end_date)
    else:
        query = f"""
            SELECT TO_CHAR(date, 'YYYY-MM-DD') as date, SUM(units) as units
            FROM {_FACT}
            WHERE store_id = %s AND date BETWEEN %s AND %s
            GROUP BY date
            ORDER BY date
        """
        params = (store_id, start_date, end_date)

    df = run_query(query, params)
    return {"data": df.to_dict(orient="records")}


@router.get("/performance")
async def get_performance(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None
):
    """Top 10 items and stores"""
    if store_id:
        top_items = run_query(
            f"SELECT item_id, SUM(units) as units FROM {_FACT} WHERE date BETWEEN %s AND %s AND store_id = %s GROUP BY item_id ORDER BY units DESC LIMIT 10",
            (start_date, end_date, store_id)
        )
    else:
        top_items = run_query(
            f"SELECT item_id, SUM(units) as units FROM {_FACT} WHERE date BETWEEN %s AND %s GROUP BY item_id ORDER BY units DESC LIMIT 10",
            (start_date, end_date)
        )

    top_stores = run_query(
        f"SELECT store_id, SUM(units) as units FROM {_FACT} WHERE date BETWEEN %s AND %s GROUP BY store_id ORDER BY units DESC LIMIT 10",
        (start_date, end_date)
    )
    return {
        "top_items": top_items.to_dict(orient="records"),
        "top_stores": top_stores.to_dict(orient="records")
    }


@router.get("/compare")
async def compare_products(
    start_date: date,
    end_date: date,
    item_ids: List[int] = Query(...),
    store_id: Optional[int] = None
):
    """Benchmark multiple products"""
    item_list = ",".join(map(str, item_ids))
    if store_id:
        query = f"""
            SELECT TO_CHAR(date, 'YYYY-MM-DD') as date, item_id, units
            FROM {_FACT}
            WHERE item_id IN ({item_list}) AND date BETWEEN %s AND %s AND store_id = %s
        """
        params = (start_date, end_date, store_id)
    else:
        query = f"""
            SELECT TO_CHAR(date, 'YYYY-MM-DD') as date, item_id, units
            FROM {_FACT}
            WHERE item_id IN ({item_list}) AND date BETWEEN %s AND %s
        """
        params = (start_date, end_date)

    df = run_query(query, params)
    return {"data": df.to_dict(orient="records")}


@router.get("/distribution")
async def get_distribution(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None,
    item_id: Optional[int] = None
):
    """Get sales distribution data (Histogram)"""
    conditions = ["date BETWEEN %s AND %s"]
    params_list: List[Any] = [start_date, end_date]

    if store_id:
        conditions.append("store_id = %s")
        params_list.append(store_id)
    if item_id:
        conditions.append("item_id = %s")
        params_list.append(item_id)

    where = " AND ".join(conditions)
    bound = tuple(params_list)

    df1 = run_query(f"SELECT units FROM {_FACT} WHERE {where} LIMIT 10000", bound)
    df2 = run_query(
        f"SELECT store_id, TO_CHAR(date, 'YYYY-MM-DD') as date, SUM(units) as store_day_units FROM {_FACT} WHERE {where} GROUP BY store_id, date",
        bound
    )

    return {
        "detailed": df1["units"].tolist() if not df1.empty else [],
        "aggregated": df2["store_day_units"].tolist() if not df2.empty else []
    }
