from fastapi import APIRouter, status, Depends
from src.api.dependencies import get_model_manager
from src.core.model import ModelManager

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "Sales Forecasting API"}

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(manager: ModelManager = Depends(get_model_manager)):
    """Readiness probe - check if model is loaded."""
    model_loaded = manager.model is not None
    return {
        "status": "ready" if model_loaded else "not_ready",
        "model_loaded": model_loaded
    }