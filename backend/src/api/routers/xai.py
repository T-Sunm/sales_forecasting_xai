"""
XAI (Explainable AI) Router
Endpoints for SHAP-based model explanations
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

from src.core.model import ModelManager
from src.core import xai_explainer
from src.api.dependencies import get_model_manager, get_feature_data
from src.config import COL_STORE_ID, COL_ITEM_ID

router = APIRouter(prefix="/xai", tags=["XAI"])


# ==================== REQUEST/RESPONSE MODELS ====================

class LocalExplanationRequest(BaseModel):
    """Request body for local explanation"""
    date: Optional[str] = Field(None, description="Date to explain (YYYY-MM-DD)")
    sample_index: Optional[int] = Field(None, description="Sample index to explain")
    top_n: int = Field(10, description="Number of top features to return")


# ==================== ENDPOINTS ====================

@router.post("/global_importance")
async def get_global_importance(
    store_id: int = Query(..., description="Store ID"),
    item_id: int = Query(..., description="Item ID"),
    sample_size: int = Query(500, description="Number of samples for SHAP computation"),
    manager: ModelManager = Depends(get_model_manager),
    feature_data: pd.DataFrame = Depends(get_feature_data)
):
    """
    Compute global SHAP feature importance for a store-item pair
    
    Returns:
        dict: {
            "store_id": int,
            "item_id": int,
            "feature_importance": List[{"feature": str, "importance": float, "category": str}],
            "category_summary": List[{"category": str, "importance_pct": float, "num_features": int}]
        }
    """
    # Get model
    model = manager.get_model(store_id, item_id)
    if model is None:
        raise HTTPException(404, detail=f"No model found for store {store_id}, item {item_id}")
    
    # Get data for this store-item pair
    store_item_data = feature_data[
        (feature_data[COL_STORE_ID] == store_id) &
        (feature_data[COL_ITEM_ID] == item_id)
    ]
    
    if store_item_data.empty:
        raise HTTPException(404, detail=f"No data found for store {store_id}, item {item_id}")
    
    # Sample data
    if len(store_item_data) > sample_size:
        X_sample = store_item_data.sample(n=sample_size, random_state=42)
    else:
        X_sample = store_item_data
    
    # Get feature names and prepare X
    feature_names = model.feature_name_
    X_values = X_sample[feature_names].values
    
    # Compute SHAP
    shap_values, expected_value = xai_explainer.compute_shap_values(
        model, X_values, feature_names
    )
    
    # Get importance
    feature_importance = xai_explainer.get_feature_importance(shap_values, feature_names)
    category_summary = xai_explainer.get_category_summary(feature_importance)
    
    return {
        "store_id": store_id,
        "item_id": item_id,
        "feature_importance": feature_importance,
        "category_summary": category_summary
    }


@router.post("/dependence")
async def get_dependence(
    store_id: int = Query(..., description="Store ID"),
    item_id: int = Query(..., description="Item ID"),
    feature: str = Query(..., description="Feature name to analyze"),
    sample_size: int = Query(500, description="Number of samples"),
    interaction_feature: Optional[str] = Query(None, description="Feature for interaction coloring"),
    manager: ModelManager = Depends(get_model_manager),
    feature_data: pd.DataFrame = Depends(get_feature_data)
):
    """
    Get SHAP dependence plot data for a specific feature
    
    Returns:
        dict: {
            "feature": str,
            "feature_values": List[float],
            "shap_values": List[float],
            "interaction_feature": str (optional),
            "interaction_values": List[float] (optional),
            "stats": {"mean": float, "std": float, "min": float, "max": float}
        }
    """
    # Get model
    model = manager.get_model(store_id, item_id)
    if model is None:
        raise HTTPException(404, detail=f"No model found for store {store_id}, item {item_id}")
    
    # Get data
    store_item_data = feature_data[
        (feature_data[COL_STORE_ID] == store_id) &
        (feature_data[COL_ITEM_ID] == item_id)
    ]
    
    if store_item_data.empty:
        raise HTTPException(404, detail=f"No data found for store {store_id}, item {item_id}")
    
    # Sample
    if len(store_item_data) > sample_size:
        X_sample = store_item_data.sample(n=sample_size, random_state=42)
    else:
        X_sample = store_item_data
    
    # Get feature names
    feature_names = model.feature_name_
    
    # Check if feature exists
    if feature not in feature_names:
        raise HTTPException(400, detail=f"Feature '{feature}' not found in model")
    
    X_values = X_sample[feature_names].values
    
    # Compute SHAP
    shap_values, _ = xai_explainer.compute_shap_values(model, X_values, feature_names)
    
    # Get dependence data
    dependence_data = xai_explainer.get_dependence_data(
        shap_values,
        X_sample[feature_names],
        feature,
        interaction_feature
    )
    
    return dependence_data


@router.post("/local_explanation")
async def get_local_explanation(
    store_id: int = Query(..., description="Store ID"),
    item_id: int = Query(..., description="Item ID"),
    manager: ModelManager = Depends(get_model_manager),
    feature_data: pd.DataFrame = Depends(get_feature_data),
    request: LocalExplanationRequest = Body(...)
):
    """
    Get SHAP waterfall explanation for a specific date or sample
    
    Request body:
        {
            "date": "2017-06-15" (optional),
            "sample_index": 100 (optional),
            "top_n": 10
        }
    
    Returns:
        dict: {
            "base_value": float,
            "prediction": float,
            "features": List[{"feature": str, "value": float, "shap_impact": float}],
            "increasing_factors": List[...],
            "decreasing_factors": List[...]
        }
    """
    # Get model
    model = manager.get_model(store_id, item_id)
    if model is None:
        raise HTTPException(404, detail=f"No model found for store {store_id}, item {item_id}")
    
    # Get data
    store_item_data = feature_data[
        (feature_data[COL_STORE_ID] == store_id) &
        (feature_data[COL_ITEM_ID] == item_id)
    ]
    
    if store_item_data.empty:
        raise HTTPException(404, detail=f"No data found for store {store_id}, item {item_id}")
    
    # Find sample index
    if request.date:
        # Filter by date
        date_data = store_item_data[store_item_data["date"] == pd.to_datetime(request.date)]
        if date_data.empty:
            raise HTTPException(404, detail=f"No data found for date {request.date}")
        sample_idx = 0
        X_sample = date_data
    elif request.sample_index is not None:
        # Use provided index
        if request.sample_index >= len(store_item_data):
            raise HTTPException(400, detail=f"Sample index {request.sample_index} out of range")
        sample_idx = request.sample_index
        X_sample = store_item_data.iloc[[sample_idx]]
    else:
        # Use most recent
        X_sample = store_item_data.sort_values("date", ascending=False).head(1)
        sample_idx = 0
    
    # Get feature names
    feature_names = model.feature_name_
    X_values = X_sample[feature_names].values
    
    # Compute SHAP
    shap_values, expected_value = xai_explainer.compute_shap_values(
        model, X_values, feature_names
    )
    
    # Get local explanation
    explanation = xai_explainer.get_local_explanation(
        shap_values,
        expected_value,
        X_sample[feature_names],
        0,  # Always 0 since we filtered to single row
        request.top_n
    )
    
    # Add actual value if available
    if "logunits" in X_sample.columns:
        explanation["actual"] = xai_explainer.safe_float(X_sample.iloc[0]["logunits"])
    
    return explanation
