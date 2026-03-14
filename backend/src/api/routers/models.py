"""
Models Router
Shared endpoints for model information (used by both Prediction & XAI)
"""

from fastapi import APIRouter, Depends
import pandas as pd
from src.core.model import ModelManager
from src.api.dependencies import get_model_manager, get_feature_data
from src.config import COL_STORE_ID, COL_ITEM_ID

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/available")
async def get_available_models(
    manager: ModelManager = Depends(get_model_manager),
    data: pd.DataFrame = Depends(get_feature_data)
):
    """Get list of available store/item pairs that can be analyzed."""
    if manager.model is None:
        return {"available": False, "models": [], "count": 0}
    
    # Get unique store/item pairs from data
    pairs = data[[COL_STORE_ID, COL_ITEM_ID]].drop_duplicates()
    models_list = [
        {"store_id": int(row[COL_STORE_ID]), "item_id": int(row[COL_ITEM_ID])}
        for _, row in pairs.iterrows()
    ]
    
    return {
        "available": True,
        "models": models_list,
        "count": len(models_list)
    }
