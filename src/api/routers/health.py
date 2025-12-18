from fastapi import APIRouter, status, Depends
from src.api.dependencies import get_model_manager
from src.core.model import ModelManager

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Liveness probe: Kiểm tra server có đang chạy không.
    """
    return {"status": "ok", "service": "Sales Forecasting API"}

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Readiness probe: Kiểm tra Models đã load thành công chưa.
    Nếu dependency 'get_model_manager' fail (raise 503), endpoint này cũng sẽ trả về 503.
    """
    model_count = len(manager.models_dict) if manager.models_dict else 0
    return {
        "status": "ready",
        "models_loaded": model_count
    }