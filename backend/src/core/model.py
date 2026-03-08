"""
Core Model Module - Business Logic Layer
Separates prediction logic from UI layer for better testability and reusability.
"""

import mlflow
import mlflow.pyfunc
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_MODEL_ALIAS,
    COL_STORE_ID,
    COL_ITEM_ID,
)


# ==================== PYDANTIC MODELS ====================

class PredictionInput(BaseModel):
    """Validated input structure for sales prediction"""
    # Date information
    date: date
    year: int
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    season: str
    
    # Holiday flags
    is_holiday: int = Field(ge=0, le=1)
    is_blackfriday: int = Field(ge=0, le=1, default=0)
    
    # Weather features
    tmax: float
    cool: float
    preciptotal: float
    stnpressure: float
    sealevel: float
    resultspeed: float
    resultdir: float
    
    # Weather codes (optional, defaults to 0)
    BCFG: int = Field(default=0, ge=0, le=1)
    BLDU: int = Field(default=0, ge=0, le=1)
    BLSN: int = Field(default=0, ge=0, le=1)
    BR: int = Field(default=0, ge=0, le=1)
    DU: int = Field(default=0, ge=0, le=1)
    DZ: int = Field(default=0, ge=0, le=1)
    FG: int = Field(default=0, ge=0, le=1)
    FU: int = Field(default=0, ge=0, le=1)
    FZDZ: int = Field(default=0, ge=0, le=1)
    FZFG: int = Field(default=0, ge=0, le=1)
    FZRA: int = Field(default=0, ge=0, le=1)
    GR: int = Field(default=0, ge=0, le=1)
    GS: int = Field(default=0, ge=0, le=1)
    HZ: int = Field(default=0, ge=0, le=1)
    MIFG: int = Field(default=0, ge=0, le=1)
    PL: int = Field(default=0, ge=0, le=1)
    PRFG: int = Field(default=0, ge=0, le=1)
    RA: int = Field(default=0, ge=0, le=1)
    SG: int = Field(default=0, ge=0, le=1)
    SN: int = Field(default=0, ge=0, le=1)
    SQ: int = Field(default=0, ge=0, le=1)
    TS: int = Field(default=0, ge=0, le=1)
    TSRA: int = Field(default=0, ge=0, le=1)
    TSSN: int = Field(default=0, ge=0, le=1)
    UP: int = Field(default=0, ge=0, le=1)
    VCFG: int = Field(default=0, ge=0, le=1)
    VCTS: int = Field(default=0, ge=0, le=1)
    
    @validator('season')
    def validate_season(cls, v):
        allowed = ['Spring', 'Summer', 'Fall', 'Winter']
        if v not in allowed:
            raise ValueError(f'Season must be one of {allowed}')
        return v
    
    class Config:
        arbitrary_types_allowed = True


class PredictionOutput(BaseModel):
    """Validated output structure for sales prediction"""
    prediction_value: float
    store_id: int
    item_id: int
    prediction_date: date
    forecast_history: Optional[pd.DataFrame] = Field(default=None, exclude=True)
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


# ==================== MODEL MANAGER CLASS ====================

class ModelManager:
    """
    Manages LightGBM model and prediction logic.
    Uses a single global model with store_id/item_id as categorical features.
    """
    
    def __init__(self):
        self.model: Optional[Any] = None
        self._model_impl = None

    def load_models(self) -> bool:
        """
        Load trained model from MLflow Model Registry via @champion alias.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            # Extract raw LightGBM booster for SHAP compatibility
            self._model_impl = self.model._model_impl.lgb_model
            return True
        except Exception:
            return False
    
    def get_feature_names(self) -> list:
        """Get feature names from the underlying LightGBM model."""
        return self._model_impl.feature_name_

    def get_raw_model(self):
        """
        Extract the underlying LightGBM booster from MLflow PyFunc wrapper.
        Required for SHAP TreeExplainer which cannot work with PyFunc wrappers.
        """
        return self._model_impl
    
    def get_model(self, store_id: int = None, item_id: int = None):
        """
        Retrieve the global model.
        store_id and item_id params kept for API compatibility.
        """
        return self.model

    def prepare_input(
        self,
        recent_samples: pd.DataFrame,
        prediction_input: PredictionInput
    ) -> pd.Series:
        """
        Prepare feature vector for prediction.
        Combines historical features (lag, rolling stats, EWMA) with user inputs.
        
        Args:
            recent_samples: Recent historical data for this store-item pair
            prediction_input: Validated user input (Pydantic model)
            
        Returns:
            pd.Series: Complete feature vector ready for model prediction
        """
        # Start with most recent sample to preserve lag features
        input_row = recent_samples.iloc[0].copy()
        
        # Convert Pydantic model to dict
        user_data = prediction_input.model_dump()
        
        # Update date-related features
        input_row["date"] = pd.to_datetime(user_data["date"])
        input_row["day"] = user_data["day"]
        input_row["month"] = user_data["month"]
        input_row["year"] = user_data["year"]
        input_row["day_of_week"] = user_data["day_of_week"]
        input_row["is_weekend"] = user_data["is_weekend"]
        
        # Update holiday features
        input_row["is_holiday"] = user_data["is_holiday"]
        if "is_blackfriday" in input_row:
            input_row["is_blackfriday"] = user_data["is_blackfriday"]
        
        # Update season features (one-hot encoding)
        for s in ["Spring", "Summer", "Winter"]:
            col_name = f"season_{s}"
            if col_name in input_row:
                input_row[col_name] = 1 if user_data["season"] == s else 0
        
        # Update weather numerical features
        weather_features = [
            "tmax", "cool", "preciptotal", 
            "stnpressure", "sealevel", "resultspeed", "resultdir"
        ]
        for feature in weather_features:
            if feature in input_row:
                input_row[feature] = user_data[feature]
        
        # Update weather code features (binary indicators)
        weather_codes = [
            'BCFG', 'BLDU', 'BLSN', 'BR', 'DU', 'DZ', 'FG', 'FU', 
            'FZDZ', 'FZFG', 'FZRA', 'GR', 'GS', 'HZ', 'MIFG', 'PL', 'PRFG', 
            'RA', 'SG', 'SN', 'SQ', 'TS', 'TSRA', 'TSSN', 'UP', 'VCFG', 'VCTS'
        ]
        for code in weather_codes:
            if code in input_row:
                input_row[code] = user_data.get(code, 0)
        
        return input_row
    
    def predict(
        self,
        store_id: int,
        item_id: int,
        prediction_input: PredictionInput,
        feature_engineered_data: pd.DataFrame,
        store_col: str = COL_STORE_ID,
        item_col: str = COL_ITEM_ID,
        use_recursive: bool = True
    ) -> PredictionOutput:
        """
        Generate sales prediction for a specific store-item pair.
        
        Args:
            store_id: Store identifier
            item_id: Item identifier
            prediction_input: Validated user input
            feature_engineered_data: Full feature-engineered dataset
            store_col: Column name for store identifier
            item_col: Column name for item identifier
            use_recursive: Whether to use recursive forecasting for future dates
            
        Returns:
            PredictionOutput: Prediction result with metadata
        """
        # Get specific model
        model = self.get_model(store_id, item_id)
        if model is None:
            return PredictionOutput(
                prediction_value=0.0,
                store_id=store_id,
                item_id=item_id,
                prediction_date=prediction_input.date,
                error=f"No trained model found for Store {store_id} and Item {item_id}"
            )
        
        # Get recent historical data for this store-item pair
        recent_samples = (
            feature_engineered_data[
                (feature_engineered_data[store_col] == store_id)
                & (feature_engineered_data[item_col] == item_id)
            ]
            .sort_values("date", ascending=False)
            .head(5)
        )
        
        if recent_samples.empty:
            return PredictionOutput(
                prediction_value=0.0,
                store_id=store_id,
                item_id=item_id,
                prediction_date=prediction_input.date,
                error="No historical data found for this store-item combination"
            )
        
        # Check if we need recursive forecasting
        last_historical_date = recent_samples['date'].max()
        target_date_pd = pd.to_datetime(prediction_input.date)
        days_gap = (target_date_pd - last_historical_date).days
        
        forecast_history = None
        
        if days_gap > 1 and use_recursive:
            # Use recursive forecasting (now from core layer)
            from core.forecasting import recursive_forecast
            
            final_prediction, forecast_history = recursive_forecast(
                feature_engineered_data=feature_engineered_data,
                model=model,
                store_id=store_id,
                item_id=item_id,
                store_col=store_col,
                item_col=item_col,
                target_date=target_date_pd,
                prediction_inputs=prediction_input.model_dump(),
                max_forecast_days=5000
            )
            prediction_units = final_prediction
            
        else:
            # Direct prediction for dates within historical range
            input_row = self.prepare_input(recent_samples, prediction_input)
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_row])
            
            # Select only features from model signature (generic MLflow API)
            model_features = self.get_feature_names()
            X_pred = input_df[model_features]
            
            # Make prediction (model outputs log-transformed values)
            prediction = model.predict(X_pred)[0]
            prediction_units = np.exp(prediction)
        
        return PredictionOutput(
            prediction_value=prediction_units,
            store_id=store_id,
            item_id=item_id,
            prediction_date=prediction_input.date,
            forecast_history=forecast_history,
            error=None
        )
