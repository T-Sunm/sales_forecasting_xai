from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
import pandas as pd
from datetime import date

from src.utils.db_manager import run_query

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/filters")
async def get_dashboard_filters():
    """Get initial filter values (date range, stores)"""
    date_bounds = run_query("SELECT MIN(date) as min_date, MAX(date) as max_date FROM mart_date_sales")
    if date_bounds.empty:
        raise HTTPException(503, "Database not available")

    min_date = str(date_bounds["min_date"].iloc[0])
    max_date = str(date_bounds["max_date"].iloc[0])

    stores_df = run_query("SELECT DISTINCT store_id FROM mart_store_day ORDER BY store_id")
    stores = stores_df["store_id"].tolist()

    return {"min_date": min_date, "max_date": max_date, "stores": stores}


@router.get("/items")
async def get_items(store_id: Optional[int] = None):
    """Get items, optionally filtered by store"""
    if store_id is None:
        items_df = run_query("SELECT DISTINCT item_id FROM mart_sales_base ORDER BY item_id")
    else:
        items_df = run_query(
            "SELECT DISTINCT item_id FROM mart_sales_base WHERE store_id = ? ORDER BY item_id",
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
        grain_table = "mart_date_sales"
        where_clause = "date BETWEEN ? AND ?"
        base_params = (str(start_date), str(end_date))
    else:
        grain_table = "mart_store_day"
        where_clause = "store_id = ? AND date BETWEEN ? AND ?"
        base_params = (store_id, str(start_date), str(end_date))

    dates = run_query(
        f"SELECT MIN(date) as mn, MAX(date) as mx FROM {grain_table} WHERE {where_clause}",
        base_params
    )
    if dates.empty or dates.iloc[0, 0] is None:
        return {"total_units": 0, "avg_daily": 0, "growth_pct": 0}

    min_d, max_d = dates.iloc[0, 0], dates.iloc[0, 1]
    mid_d = str(min_d + (max_d - min_d) / 2)

    if store_id is None:
        kpi_params = (mid_d, str(start_date), str(end_date))
        query = f"""
            SELECT
                SUM(total_units) as total_units,
                AVG(total_units) as avg_daily,
                SUM(sales_records) as total_records,
                SUM(CASE WHEN date <= ? THEN total_units ELSE 0 END) as p1_units,
                SUM(CASE WHEN date >  ? THEN total_units ELSE 0 END) as p2_units
            FROM {grain_table}
            WHERE date BETWEEN ? AND ?
        """
        kpi_params = (mid_d, mid_d, str(start_date), str(end_date))
        count_params = base_params
    else:
        query = f"""
            SELECT
                SUM(total_units) as total_units,
                AVG(total_units) as avg_daily,
                SUM(sales_records) as total_records,
                SUM(CASE WHEN date <= ? THEN total_units ELSE 0 END) as p1_units,
                SUM(CASE WHEN date >  ? THEN total_units ELSE 0 END) as p2_units
            FROM {grain_table}
            WHERE store_id = ? AND date BETWEEN ? AND ?
        """
        kpi_params = (mid_d, mid_d, store_id, str(start_date), str(end_date))
        count_params = base_params

    kpi_df = run_query(query, kpi_params)
    if kpi_df.empty:
        return {"total_units": 0, "avg_daily": 0}

    row = kpi_df.iloc[0]
    p1 = row["p1_units"] or 0
    p2 = row["p2_units"] or 0
    growth = ((p2 - p1) / p1 * 100) if p1 > 0 else 0

    days_df = run_query(
        f"SELECT COUNT(DISTINCT date) as cnt FROM {grain_table} WHERE {where_clause}",
        count_params
    )
    days_count = days_df.iloc[0, 0] or 1

    return {
        "total_units": float(row["total_units"] or 0),
        "avg_daily": float(row["avg_daily"] or 0),
        "total_records": int(row["total_records"] or 0),
        "growth_pct": float(growth),
        "days_count": int(days_count),
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
        query = "SELECT CAST(date AS VARCHAR) as date, total_units as units FROM mart_date_sales WHERE date BETWEEN ? AND ? ORDER BY date"
        params = (str(start_date), str(end_date))
    else:
        query = "SELECT CAST(date AS VARCHAR) as date, total_units as units FROM mart_store_day WHERE store_id = ? AND date BETWEEN ? AND ? ORDER BY date"
        params = (store_id, str(start_date), str(end_date))

    df = run_query(query, params)
    if df.empty:
        return {"data": []}

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
            "SELECT item_id, SUM(units) as units FROM mart_sales_base WHERE date BETWEEN ? AND ? AND store_id = ? GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
            (str(start_date), str(end_date), store_id)
        )
    else:
        top_items = run_query(
            "SELECT item_id, SUM(units) as units FROM mart_sales_base WHERE date BETWEEN ? AND ? GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
            (str(start_date), str(end_date))
        )

    top_stores = run_query(
        "SELECT store_id, SUM(total_units) as units FROM mart_store_day WHERE date BETWEEN ? AND ? GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
        (str(start_date), str(end_date))
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
    # Trino không hỗ trợ bind array, dùng IN (...) với int ids an toàn
    item_list = ",".join(map(str, item_ids))
    if store_id:
        query = f"""
            SELECT CAST(date AS VARCHAR) as date, item_id, units
            FROM mart_sales_base
            WHERE item_id IN ({item_list}) AND date BETWEEN ? AND ? AND store_id = ?
        """
        params = (str(start_date), str(end_date), store_id)
    else:
        query = f"""
            SELECT CAST(date AS VARCHAR) as date, item_id, units
            FROM mart_sales_base
            WHERE item_id IN ({item_list}) AND date BETWEEN ? AND ?
        """
        params = (str(start_date), str(end_date))

    df = run_query(query, params)
    if df.empty:
        return {"data": []}

    return {"data": df.to_dict(orient="records")}


@router.get("/distribution")
async def get_distribution(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None,
    item_id: Optional[int] = None
):
    """Get sales distribution data (Histogram)"""
    # Build detailed distribution query
    conditions = ["date BETWEEN ? AND ?"]
    params1 = [str(start_date), str(end_date)]

    if store_id:
        conditions.append("store_id = ?")
        params1.append(store_id)
    if item_id:
        conditions.append("item_id = ?")
        params1.append(item_id)

    where = " AND ".join(conditions)
    df1 = run_query(
        f"SELECT units FROM mart_sales_base WHERE {where} LIMIT 10000",
        tuple(params1)
    )

    # Build aggregated distribution query
    conditions2 = ["date BETWEEN ? AND ?"]
    params2 = [str(start_date), str(end_date)]

    if store_id:
        conditions2.append("store_id = ?")
        params2.append(store_id)
    if item_id:
        conditions2.append("item_id = ?")
        params2.append(item_id)

    where2 = " AND ".join(conditions2)
    df2 = run_query(
        f"SELECT store_id, CAST(date AS VARCHAR) as date, SUM(units) as store_day_units FROM mart_sales_base WHERE {where2} GROUP BY 1, 2",
        tuple(params2)
    )

    return {
        "detailed": df1["units"].tolist() if not df1.empty else [],
        "aggregated": df2["store_day_units"].tolist() if not df2.empty else []
    }
