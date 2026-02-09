from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import date, timedelta
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
    
    return {
        "min_date": min_date,
        "max_date": max_date,
        "stores": stores
    }

@router.get("/items")
async def get_items(store_id: Optional[int] = None):
    """Get items, optionally filtered by store"""
    if store_id is None:
        query = "SELECT DISTINCT item_id FROM mart_sales_base ORDER BY item_id"
        params = {}
    else:
        query = "SELECT DISTINCT item_id FROM mart_sales_base WHERE store_id = :sid ORDER BY item_id"
        params = {"sid": store_id}
        
    items_df = run_query(query, params)
    return {"items": items_df["item_id"].tolist()}

@router.get("/kpis")
async def get_kpis(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None
):
    """Calculate KPI metrics for the given range"""
    # Calculate mid-point and grain table
    if store_id is None:
        grain_table = "mart_date_sales"
        where_clause = "date BETWEEN :start AND :end"
        params = {"start": start_date, "end": end_date}
    else:
        grain_table = "mart_store_day"
        where_clause = "store_id = :sid AND date BETWEEN :start AND :end"
        params = {"sid": store_id, "start": start_date, "end": end_date}
        
    # Get actual date range for mid-point
    dates = run_query(f"SELECT MIN(date) as mn, MAX(date) as mx FROM {grain_table} WHERE {where_clause}", params)
    if dates.empty or dates.iloc[0,0] is None:
        return {"total_units": 0, "avg_daily": 0, "growth_pct": 0}
        
    min_d, max_d = dates.iloc[0,0], dates.iloc[0,1]
    mid_d = min_d + (max_d - min_d) / 2
    params["mid"] = mid_d
    
    query = f"""
        SELECT 
            SUM(total_units) as total_units,
            AVG(total_units) as avg_daily,
            SUM(sales_records) as total_records,
            SUM(CASE WHEN date <= :mid THEN total_units ELSE 0 END) as p1_units,
            SUM(CASE WHEN date > :mid THEN total_units ELSE 0 END) as p2_units
        FROM {grain_table}
        WHERE {where_clause}
    """
    kpi_df = run_query(query, params)
    if kpi_df.empty:
        return {"total_units": 0, "avg_daily": 0}
        
    row = kpi_df.iloc[0]
    p1 = row["p1_units"] or 0
    p2 = row["p2_units"] or 0
    growth = ((p2 - p1) / p1 * 100) if p1 > 0 else 0
    
    # Day count
    days_count = run_query(f"SELECT COUNT(DISTINCT date) as cnt FROM {grain_table} WHERE {where_clause}", params).iloc[0,0] or 1
    
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
        query = "SELECT date, total_units as units FROM mart_date_sales WHERE date BETWEEN :s AND :e ORDER BY date"
        params = {"s": start_date, "e": end_date}
    else:
        query = "SELECT date, total_units as units FROM mart_store_day WHERE store_id = :sid AND date BETWEEN :s AND :e ORDER BY date"
        params = {"sid": store_id, "s": start_date, "e": end_date}
        
    df = run_query(query, params)
    if df.empty:
        return {"data": []}
        
    df["date"] = df["date"].astype(str)
    return {"data": df.to_dict(orient="records")}

@router.get("/performance")
async def get_performance(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None
):
    """Top 10 items and stores"""
    # Top Items
    q_items = "SELECT item_id, SUM(units) as units FROM mart_sales_base WHERE date BETWEEN :s AND :e"
    p_items = {"s": start_date, "e": end_date}
    if store_id:
        q_items += " AND store_id = :sid"
        p_items["sid"] = store_id
    q_items += " GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
    
    top_items = run_query(q_items, p_items)
    
    # Top Stores
    q_stores = "SELECT store_id, SUM(total_units) as units FROM mart_store_day WHERE date BETWEEN :s AND :e GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
    top_stores = run_query(q_stores, {"s": start_date, "e": end_date})
    
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
    query = """
        SELECT date, item_id, units 
        FROM mart_sales_base 
        WHERE item_id = ANY(:items) AND date BETWEEN :s AND :e
    """
    params = {"items": item_ids, "s": start_date, "e": end_date}
    if store_id:
        query += " AND store_id = :sid"
        params["sid"] = store_id
        
    df = run_query(query, params)
    if df.empty:
        return {"data": []}
        
    df["date"] = df["date"].astype(str)
    return {"data": df.to_dict(orient="records")}


@router.get("/distribution")
async def get_distribution(
    start_date: date,
    end_date: date,
    store_id: Optional[int] = None,
    item_id: Optional[int] = None
):
    """Get sales distribution data (Histogram)"""
    
    # 1. Detailed Distribution (Item-Store-Day)
    # This query gets individual sales records
    q1 = "SELECT units FROM mart_sales_base WHERE date BETWEEN :s AND :e"
    p1 = {"s": start_date, "e": end_date}
    
    if store_id:
        q1 += " AND store_id = :sid"
        p1["sid"] = store_id
    if item_id:
        q1 += " AND item_id = :iid"
        p1["iid"] = item_id
        
    q1 += " LIMIT 10000"
    df1 = run_query(q1, p1)
    
    # 2. Aggregated Distribution (Store-Day)
    # This sums up sales per store per day
    q2 = """
        SELECT store_id, date, SUM(units) as store_day_units
        FROM mart_sales_base
        WHERE date BETWEEN :s AND :e
    """
    p2 = {"s": start_date, "e": end_date}
    
    if store_id:
        q2 += " AND store_id = :sid"
        p2["sid"] = store_id
    # No item filter for aggregated store-day unless specifically analyzing store behavior for that item?
    # Usually store-day aggregation looks at total store sales.
    # BUT if we filter by item, we are looking at total sales of that item across days?
    # Let's align with logic: "Store-Day (Aggregated)" usually means how busy the store is.
    # If item is selected, it might mean "Item-Store sales per day".
    if item_id:
        q2 += " AND item_id = :iid"
        p2["iid"] = item_id
        
    q2 += " GROUP BY 1, 2"
    df2 = run_query(q2, p2)
    
    return {
        "detailed": df1["units"].tolist() if not df1.empty else [],
        "aggregated": df2["store_day_units"].tolist() if not df2.empty else []
    }
