"""
Models Router
Shared endpoints for model information (used by both Prediction & XAI)
"""

from fastapi import APIRouter, Depends
from core.model import ModelManager
from api.dependencies import get_model_manager

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/available")
async def get_available_models(manager: ModelManager = Depends(get_model_manager)):
    """
    Get list of (store_id, item_id) pairs that have trained models.
    
    Used by:
    - Prediction UI: To show which store-item combinations can be predicted
    - XAI UI: To populate store/item selectors for analysis
    
    Returns:
        dict: {
            "models": [{"store_id": int, "item_id": int}, ...],
            "count": int
        }
    """
    if manager.models_dict is None:
        return {"models": [], "count": 0}
    
    # Convert model keys to list of dicts
    models = [
        {"store_id": int(store_id), "item_id": int(item_id)} 
        for (store_id, item_id) in manager.models_dict.keys()
    ]
    
    return {
        "models": models,
        "count": len(models)
    }
