"""
[DEPRECATED] UI wrapper for recursive forecasting with Streamlit integration.

This file is NO LONGER USED after Frontend-Backend split.
Backend API (/prediction/predict) now handles recursive forecasting internally.

Kept for reference only. Will be removed in future cleanup.
"""

# DEPRECATED - Backend handles this now
# from src.core.forecasting import recursive_forecast as core_recursive_forecast

# def recursive_forecast_ui(*args, **kwargs):
#     """
#     [DEPRECATED] Streamlit wrapper for recursive_forecast from Core layer.
#     
#     Backend API endpoint /prediction/predict now handles:
#     - Recursive day-by-day forecasting
#     - Feature preparation with lag/rolling/EWMA
#     - Progress tracking (on backend side)
#     
#     No longer needed in frontend.
#     """
#     pass

