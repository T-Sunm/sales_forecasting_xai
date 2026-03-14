from fastapi import APIRouter, Request, status

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "Sales Forecasting API"}

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(request: Request):
    """Readiness probe - check if model is loaded."""
    manager = getattr(request.app.state, "model_manager", None)
    model_loaded = manager is not None and manager.model is not None
    return {
        "status": "ready" if model_loaded else "not_ready",
        "model_loaded": model_loaded,
        "models_loaded": 1 if model_loaded else 0,
    }