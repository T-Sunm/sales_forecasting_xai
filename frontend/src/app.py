import streamlit as st
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from services import get_api_client
from utils.local_data_loader import (
    load_data,
    load_feature_engineered_data,
    load_feature_stats,
)
from components.ui_builder.dashboard import historical_sales_view
from components.ui_predictor.prediction import sales_prediction_view
from components.ui_xai.xai_view import xai_explanation_view

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting XAI", 
    page_icon="📈", 
    layout="wide"
)


def check_backend_connection():
    """Check if backend API is available"""
    api = get_api_client()
    
    with st.spinner("Connecting to backend..."):
        if not api.is_backend_available():
            st.error(
                "⚠️ **Backend server is not running!**\n\n"
                "Please start the backend server first:\n"
                "```bash\n"
                "cd backend\n"
                "uv run python run.py\n"
                "```"
            )
            st.info("Backend should be running at http://localhost:8000")
            st.stop()
        
        # Check if models are loaded
        ready = api.readiness_check()
        if "error" in ready:
            st.warning(
                "⚠️ Backend is running but models are not loaded properly.\n"
                "Prediction features may not work."
            )
        else:
            models_count = ready.get("models_loaded", 0)
            st.sidebar.success(f"✅ Backend connected ({models_count:,} models)")


def main():
    # Sidebar title
    st.sidebar.title("Sales Forecasting XAI")
    
    page = st.sidebar.selectbox(
        "Choose a page", 
        [
            "Historical Sales Analysis", 
            "Sales Prediction", 
            "XAI Explanation"
        ]
    )
    
    # Page logic
    if page == "Historical Sales Analysis":
        # Direct SQL mode - no heavy CSV loading needed
        historical_sales_view()
        
    elif page == "Sales Prediction":
        check_backend_connection()
        # Load data only when needed
        data = load_data()
        feature_stats = load_feature_stats()
        feature_engineered_data = load_feature_engineered_data()
        api = get_api_client()
        sales_prediction_view(data, api, feature_stats, feature_engineered_data)
        
    else:  # XAI Explanation
        check_backend_connection()
        # Load data only when needed
        data = load_data()
        # XAI now uses Backend API for all SHAP computations
        xai_explanation_view(data, None, None)


if __name__ == "__main__":
    main()