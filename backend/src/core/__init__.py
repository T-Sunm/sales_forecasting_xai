"""
Core module - Business Logic Layer
Contains model management and prediction logic separated from UI.
"""

from src.core.model import ModelManager, PredictionInput, PredictionOutput
from src.core.forecasting import recursive_forecast, get_historical_weather_average

__all__ = [
    'ModelManager', 
    'PredictionInput', 
    'PredictionOutput',
    'recursive_forecast',
    'get_historical_weather_average'
]
