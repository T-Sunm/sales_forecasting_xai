from fastapi import Request, HTTPException, status
import pandas as pd
from src.core.model import ModelManager

async def get_model_manager(request: Request) -> ModelManager:
    """
    Dependency to retrieve the initialized ModelManager from app state.
    Ensures safe access and handles cases where models aren't loaded.
    
    Usage in router:
        @router.post("/predict")
        async def predict(
            input_data: PredictionInput,
            manager: ModelManager = Depends(get_model_manager)
        ):
            ...
    
    Raises:
        HTTPException(503): If ModelManager is not initialized or models aren't loaded.
    """
    model_manager = getattr(request.app.state, "model_manager", None)
    
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model Manager is not initializing. Please wait a moment."
        )

    if model_manager.models_dict is None:
         raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive models are not loaded. Cannot process request."
        )
        
    return model_manager


async def get_feature_data(request: Request) -> pd.DataFrame:
    """
    Dependency to retrieve the loaded feature engineering data.
    Required for prediction context (historical features, weather, etc.)
    """
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
