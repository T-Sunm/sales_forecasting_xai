"""
Data Router
Endpoints for serving data to frontend (JSON format for HTML/CSS/JS compatibility)
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
import pandas as pd

from src.api.dependencies import get_feature_data
from src.config import COL_STORE_ID, COL_ITEM_ID

router = APIRouter(prefix="/data", tags=["Data"])


@router.get("/historical")
async def get_historical_data(
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    item_id: Optional[int] = Query(None, description="Filter by item ID"),
    limit: int = Query(1000, description="Max records to return", le=10000),
    feature_data: pd.DataFrame = Depends(get_feature_data)
) -> dict:
    """
    Get historical sales data as JSON
    
    Returns:
        dict: {
            "data": List of records,
            "total_records": int,
            "columns": List of column names
        }
    """
    df = feature_data.copy()
    
    if store_id is not None:
        df = df[df[COL_STORE_ID] == store_id]
    
    if item_id is not None:
        df = df[df[COL_ITEM_ID] == item_id]
    
    if df.empty:
        return {"data": [], "total_records": 0, "columns": []}
    
    df = df.sort_values("date", ascending=False).head(limit)
    
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    
    return {
        "data": df.to_dict(orient="records"),
        "total_records": len(df),
        "columns": list(df.columns)
    }


@router.get("/top_pair")
async def get_top_data_pair(
    feature_data: pd.DataFrame = Depends(get_feature_data)
) -> dict:
    """
    Find the (store_id, item_id) pair with the most records
    
    Returns:
        dict: {"store_id": int, "item_id": int, "record_count": int}
    """
    if feature_data.empty:
        raise HTTPException(404, detail="No data available")
    
    counts = (
        feature_data.groupby([COL_STORE_ID, COL_ITEM_ID])
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )
    
    if counts.empty:
        raise HTTPException(404, detail="No store-item pairs found")
    
    top_row = counts.iloc[0]
    return {
        "store_id": int(top_row[COL_STORE_ID]),
        "item_id": int(top_row[COL_ITEM_ID]),
        "record_count": int(top_row["n_rows"])
    }


@router.get("/metadata")
async def get_data_metadata(
    feature_data: pd.DataFrame = Depends(get_feature_data)
) -> dict:
    """
    Get metadata about the loaded feature data
    """
    date_min = str(feature_data["date"].min()) if "date" in feature_data.columns else None
    date_max = str(feature_data["date"].max()) if "date" in feature_data.columns else None
    
    return {
        "total_records": len(feature_data),
        "columns": list(feature_data.columns),
        "date_range": {"min": date_min, "max": date_max},
        "unique_stores": int(feature_data[COL_STORE_ID].nunique()) if COL_STORE_ID in feature_data.columns else 0,
        "unique_items": int(feature_data[COL_ITEM_ID].nunique()) if COL_ITEM_ID in feature_data.columns else 0
    }


@router.get("/stores")
async def get_stores(
    feature_data: pd.DataFrame = Depends(get_feature_data)
) -> dict:
    """
    Get list of all unique store IDs
    """
    stores = sorted(feature_data[COL_STORE_ID].unique().tolist())
    return {"stores": stores, "count": len(stores)}


@router.get("/items/{store_id}")
async def get_items_by_store(
    store_id: int,
    feature_data: pd.DataFrame = Depends(get_feature_data)
) -> dict:
    """
    Get list of unique item IDs for a specific store
    """
    items = sorted(feature_data[feature_data[COL_STORE_ID] == store_id][COL_ITEM_ID].unique().tolist())
    if not items:
        raise HTTPException(status_code=404, detail=f"No items found for store {store_id}")
    return {"store_id": store_id, "items": items, "count": len(items)}

