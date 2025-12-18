"""
UI wrapper for recursive forecasting with Streamlit integration.
This file bridges Core layer (pure logic) and UI layer (Streamlit).
"""

import streamlit as st
from src.core.forecasting import recursive_forecast as core_recursive_forecast


def recursive_forecast_ui(*args, **kwargs):
    """
    Streamlit wrapper for recursive_forecast from Core layer.
    Adds progress bar and info messages for better UX.
    """
    # Extract parameters
    target_date = kwargs.get('target_date')
    feature_engineered_data = kwargs.get('feature_engineered_data')
    store_col = kwargs.get('store_col')
    item_col = kwargs.get('item_col')
    store_id = kwargs.get('store_id')
    item_id = kwargs.get('item_id')
    
    # Get last historical date for info display
    historical_data = feature_engineered_data[
        (feature_engineered_data[store_col] == store_id) &
        (feature_engineered_data[item_col] == item_id)
    ]
    last_historical_date = historical_data['date'].max()
    days_to_forecast = (target_date - last_historical_date).days
    
    # Display info message
    st.info(
        f"🔄 Performing recursive forecasting for {days_to_forecast} days "
        f"from {last_historical_date.date()} to {target_date.date()}"
    )
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    def progress_callback(current, total):
        """Update Streamlit progress bar"""
        progress_bar.progress(current / total)
    
    # Add progress callback to kwargs
    kwargs['progress_callback'] = progress_callback
    
    # Call core function
    result = core_recursive_forecast(*args, **kwargs)
    
    # Clear progress bar
    progress_bar.empty()
    
    return result
