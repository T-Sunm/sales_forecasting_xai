from fastapi import Request, HTTPException, status
import pandas as pd
from src.core.model import ModelManager

async def get_model_manager(request: Request) -> ModelManager:
    """Retrieve ModelManager from app state. Raises 503 if not available."""
    model_manager = getattr(request.app.state, "model_manager", None)
    
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model Manager is not initializing. Please wait a moment."
        )

    if model_manager.model is None:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive model is not loaded. Cannot process request."
        )
        
    return model_manager


async def get_feature_data(request: Request) -> pd.DataFrame:
    """Retrieve feature data from app state. Raises 503 if not available."""
    feature_data = getattr(request.app.state, "feature_data", None)
    
    if feature_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature data is not loaded. Service is analyzing history..."
        )
        
    if feature_data.empty:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature data is empty. Cannot process prediction."
        )
        
    return feature_data
