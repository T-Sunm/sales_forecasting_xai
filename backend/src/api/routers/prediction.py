from fastapi import APIRouter, Depends, HTTPException
from core.model import ModelManager, PredictionInput, PredictionOutput
from api.dependencies import get_model_manager, get_feature_data
import pandas as pd

router = APIRouter(prefix="/prediction", tags=["Prediction"])

# 1. GET /stores
@router.get("/stores")
async def get_stores(data: pd.DataFrame = Depends(get_feature_data)):
    """Trả về danh sách tất cả Store ID có trong dữ liệu."""
    stores = sorted(data["store_nbr"].unique().tolist())
    return {"stores": stores, "count": len(stores)}

# 2. GET /items/{store_id}
@router.get("/items/{store_id}")
async def get_items(store_id: int, data: pd.DataFrame = Depends(get_feature_data)):
    """Trả về danh sách Item ID bán tại Store cụ thể."""
    # Lọc store
    store_data = data[data["store_nbr"] == store_id]
    if store_data.empty:
        raise HTTPException(404, detail=f"Store public ID {store_id} not found")
        
    items = sorted(store_data["item_nbr"].unique().tolist())
    return {"store_id": store_id, "items": items, "count": len(items)}



# 3. POST /predict
@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    store_id: int, 
    item_id: int,
    manager: ModelManager = Depends(get_model_manager),
    data: pd.DataFrame = Depends(get_feature_data)
):
    """Core logic dự đoán."""
    result = manager.predict(
        store_id=store_id,
        item_id=item_id,
        prediction_input=input_data,
        feature_engineered_data=data
    )
    
    if result.error:
        raise HTTPException(400, detail=result.error)
        
    return result