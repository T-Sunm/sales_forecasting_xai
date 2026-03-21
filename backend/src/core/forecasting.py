"""
Recursive forecasting utilities for time series prediction.

This module provides functions to perform iterative/recursive forecasting,
which is essential when predicting far into the future where lag features
from historical data become stale.

Pure business logic - No UI dependencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def get_historical_weather_average(
    feature_engineered_data: pd.DataFrame, 
    target_date: pd.Timestamp, 
    store_col: str, 
    item_col: str, 
    store_id: int, 
    item_id: int
) -> Dict[str, float]:
    """
    Get average weather features for a specific month/day from historical data.
    
    For example, to predict Dec 10, 2025, this returns average weather 
    from all Dec 10 dates in the historical data.
    
    Args:
        feature_engineered_data: Full dataset with historical data
        target_date: The date to get weather for
        store_col, item_col: Column names for store and item
        store_id, item_id: Specific store and item IDs
    
    Returns:
        dict: Weather features averaged from historical same month/day
    """
    # Filter to same store-item
    store_item_data = feature_engineered_data[
        (feature_engineered_data[store_col] == store_id) &
        (feature_engineered_data[item_col] == item_id)
    ].copy()
    
    # Extract month and day from target
    target_month = target_date.month
    target_day = target_date.day
    
    # Filter to same month/day from historical data
    store_item_data['month_day'] = store_item_data['date'].dt.month.astype(str) + '_' + store_item_data['date'].dt.day.astype(str)
    target_month_day = f"{target_month}_{target_day}"
    
    historical_same_day = store_item_data[store_item_data['month_day'] == target_month_day]
    
    # Weather features to average
    weather_features = [
        "tmax", "tmin", "tavg", "depart", "dewpoint", "wetbulb", "heat", "cool",
        "sunrise", "sunset", "snowfall", "preciptotal", "stnpressure", "sealevel",
        "resultspeed", "resultdir", "avgspeed"
    ]
    weather_codes = [
        'is_ra', 'is_sn', 'is_fg', 'is_br', 'is_up', 'is_ts', 'is_hz', 'is_dz', 'is_sq', 
        'is_fz', 'is_mi', 'is_pr', 'is_bc', 'is_bl', 'is_vc'
    ]
    
    weather_avg = {}
    
    if len(historical_same_day) > 0:
        # Calculate averages for numerical features
        for feature in weather_features:
            if feature in historical_same_day.columns:
                weather_avg[feature] = historical_same_day[feature].mean()
        
        # For weather codes (binary), use mode (most common value)
        for code in weather_codes:
            if code in historical_same_day.columns:
                weather_avg[code] = int(historical_same_day[code].mode()[0]) if len(historical_same_day[code].mode()) > 0 else 0
    else:
        # Fallback: use overall averages if no data for this specific day
        for feature in weather_features:
            if feature in store_item_data.columns:
                weather_avg[feature] = store_item_data[feature].mean()
        for code in weather_codes:
            if code in store_item_data.columns:
                weather_avg[code] = 0
    
    return weather_avg


def recursive_forecast(
    feature_engineered_data: pd.DataFrame,
    model,
    store_id: int,
    item_id: int,
    store_col: str,
    item_col: str,
    target_date: pd.Timestamp,
    prediction_inputs: dict,
    max_forecast_days: int = 365,
    progress_callback: Optional[callable] = None
) -> Tuple[float, pd.DataFrame]:
    """
    Perform recursive/iterative forecasting from last historical date to target date.
    
    This function:
    1. Finds the last date in historical data
    2. If target_date is far in the future, predicts day-by-day
    3. Each prediction updates lag features for the next prediction
    4. Weather features are estimated using historical seasonal averages
    
    Args:
        feature_engineered_data: Full dataset
        model: Trained model
        store_id, item_id: Target store and item
        store_col, item_col: Column names
        target_date: Date to predict for
        prediction_inputs: User inputs for the target date
        max_forecast_days: Maximum number of days to forecast recursively (safety limit)
        progress_callback: Optional function(current, total) to report progress
    
    Returns:
        final_prediction: The prediction for target_date
        forecast_history: DataFrame with all intermediate predictions
    """
    # Get historical data for this store-item
    historical_data = feature_engineered_data[
        (feature_engineered_data[store_col] == store_id) &
        (feature_engineered_data[item_col] == item_id)
    ].sort_values('date').copy()
    
    if historical_data.empty:
        raise ValueError("No historical data found for this store-item combination")
    
    # Get last historical date
    last_historical_date = historical_data['date'].max()
    target_date_pd = pd.to_datetime(target_date)
    
    # Calculate days to forecast
    days_to_forecast = (target_date_pd - last_historical_date).days
    
    # If target is in the past or very close, use simple prediction
    if days_to_forecast <= 1:
        return None, None  # Signal to use simple prediction
    
    # Safety check
    if days_to_forecast > max_forecast_days:
        days_to_forecast = max_forecast_days
    
    # Initialize forecast history
    forecast_history = []
    
    # Get model features from MLflow signature (generic, works with any flavor)
    if hasattr(model, "metadata") and model.metadata.signature:
        model_features = [inp.name for inp in model.metadata.signature.inputs]
    elif hasattr(model, "feature_name_"):
        model_features = model.feature_name_
    else:
        model_features = [col for col in historical_data.columns 
                         if col not in ["sales", "units", "logunits", "date", store_col, item_col]]
    
    # Start from last historical row
    current_row = historical_data.iloc[-1].copy()
    
    # Forecast day by day
    for day_offset in range(1, days_to_forecast + 1):
        current_date = last_historical_date + pd.Timedelta(days=day_offset)
        
        # Report progress if callback provided
        if progress_callback:
            progress_callback(day_offset, days_to_forecast)
        
        # Create new row for this date
        new_row = current_row.copy()
        new_row['date'] = current_date
        
        # Update date features
        new_row['year'] = current_date.year
        new_row['month'] = current_date.month
        new_row['day'] = current_date.day
        new_row['day_of_week'] = current_date.dayofweek
        new_row['is_weekend'] = 1 if current_date.dayofweek >= 5 else 0
        new_row['quarter'] = (current_date.month - 1) // 3 + 1
        
        # Update season features
        month = current_date.month
        if month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        elif month in [12, 1, 2]:
            season = "winter"
        else:
            season = "fall"
        
        for s in ["spring", "summer", "winter", "fall"]:
            col_name = f"season_{s}"
            if col_name in new_row.index:
                new_row[col_name] = 1 if season == s else 0
        
        # Get weather features from historical average for this month/day
        weather_avg = get_historical_weather_average(
            feature_engineered_data, current_date, store_col, item_col, store_id, item_id
        )
        
        # Update weather features
        for feature, value in weather_avg.items():
            if feature in new_row.index:
                new_row[feature] = value
        
        # If this is the target date, use user-provided inputs
        if current_date.date() == target_date.date():
            # Override with user inputs
            for feature in ['tmax', 'cool', 'preciptotal', 'stnpressure', 'sealevel', 'resultspeed', 'resultdir']:
                if feature in prediction_inputs and feature in new_row.index:
                    new_row[feature] = prediction_inputs[feature]
            
            # Weather codes
            for code in weather_avg.keys():
                if code in prediction_inputs and code in new_row.index:
                    new_row[code] = prediction_inputs[code]
            
            # Holidays
            if 'is_holiday' in new_row.index:
                new_row['is_holiday'] = prediction_inputs.get('is_holiday', 0)
            if 'is_blackfriday' in new_row.index:
                new_row['is_blackfriday'] = prediction_inputs.get('is_blackfriday', 0)
        
        # Make prediction for this day
        X_pred = pd.DataFrame([new_row])[model_features]
        prediction_log = model.predict(X_pred)[0]
        prediction_units = np.exp(prediction_log)
        
        # Store forecast
        forecast_history.append({
            'date': current_date,
            'predicted_logunits': prediction_log,
            'predicted_units': prediction_units
        })
        
        # Update lag features for next iteration
        # This is crucial: use the prediction we just made to update lags
        new_row['logunits'] = prediction_log
        if 'units' in new_row.index:
            new_row['units'] = prediction_units
        
        # Update lag features (if they exist in the model)
        for lag in [1, 2, 3, 7, 14, 28]:  # Common lag values
            lag_col = f'logunits_lag_{lag}'
            if lag_col in new_row.index:
                # Shift: lag_1 becomes what was lag_0 (current prediction)
                if lag == 1:
                    new_row[lag_col] = prediction_log
                else:
                    # Get from previous lag
                    prev_lag_col = f'logunits_lag_{lag-1}'
                    if prev_lag_col in current_row.index:
                        new_row[lag_col] = current_row[prev_lag_col]
        
        # Update rolling features (simplified - just shift)
        # In practice, you'd recalculate these properly
        for window in [7, 14, 28]:
            for stat in ['mean', 'min', 'max', 'std']:
                col_name = f'logunits_{stat}_{window}d'
                if col_name in new_row.index:
                    # Keep the value (in real implementation, recalculate from history)
                    pass
        
        # Update current row for next iteration
        current_row = new_row
    
    # Return final prediction and history
    forecast_df = pd.DataFrame(forecast_history)
    final_prediction = forecast_df.iloc[-1]['predicted_units']
    
    return final_prediction, forecast_df
