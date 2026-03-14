"""
XAI View Module
Main Streamlit view for Explainable AI dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import services
from services import get_api_client

# Import utilities
from .utils import (
    XAIStateManager,
    create_store_item_selector,
    init_llm_generator
)

# Import components
from .components import (
    display_global_explanations,
    display_dependence_analysis,
    display_local_explanations
)


def xai_explanation_view():
    """
    Main XAI view displaying SHAP-based explanations
    Components now fetch data via Backend API
    """
    st.title("🔍 Explainable AI (XAI) Dashboard")
    st.markdown("""
    Understand **how** and **why** the model makes predictions using SHAP 
    (SHapley Additive exPlanations) analysis side-by-side with backend API.
    """)
    
    # Initialize API Client
    api_client = get_api_client()
    if not api_client.is_backend_available():
        st.error("❌ Backend API is not available. Please start the backend server.")
        return
    
    # Get available models from API
    with st.spinner("Fetching available models..."):
        response = api_client.get_available_models()
        models_list = response.get("models", []) if response else []
        
    if not models_list:
        st.error("No models available for XAI analysis.")
        return
        
    # Sidebar: Store and Item selection
    store_id, item_id = create_store_item_selector(models_list)
    
    if store_id is None or item_id is None:
        st.warning("⚠️ Please select both Store and Item from sidebar")
        return
    
    # Initialize LLM (if API key available)
    llm_generator = init_llm_generator()
    
    # Reset state if selection changed
    XAIStateManager.reset_on_selection_change(store_id, item_id)
    
    # === Section 1: Global Explanations ===
    st.markdown("---")
    
    # Fetch global importance data
    with st.spinner('Calculating global feature importance (Backend)...'):
        global_data = api_client.get_global_importance(store_id, item_id, sample_size=500)
    
    if global_data:
        # Construct DataFrames expected by components
        feature_importance = global_data.get("feature_importance", [])
        category_summary = global_data.get("category_summary", [])
        
        importance_df = pd.DataFrame(feature_importance)
        category_summary_df = pd.DataFrame(category_summary)
        
        # Display global section
        display_global_explanations(
            shap_values=None,  # Not available from API
            X_sample=None,     # Not available from API
            feature_names=importance_df['feature'].tolist() if not importance_df.empty else [],
            importance_df=importance_df,
            category_summary=category_summary_df,
            store_id=store_id,
            item_id=item_id,
            llm_generator=llm_generator
        )
    else:
        st.error("Failed to load global explanation data.")
        return

    # === Section 2: Feature Dependency Analysis ===
    st.markdown("---")
    
    # Need to pass api_client to dependence analysis since it needs to fetch data repeatedly
    # based on user selection
    display_dependence_analysis(
        shap_values=None,
        X_sample=None,
        importance_df=importance_df,
        llm_generator=llm_generator,
        store_id=store_id,
        item_id=item_id,
        api_client=api_client  # New argument
    )
    
    # === Section 3: Local (Instance) Explanations ===
    st.markdown("---")
    
    display_local_explanations(
        shap_values=None,
        expected_value=None,
        X_sample=None,
        df_sample=None,
        feature_names=importance_df['feature'].tolist() if not importance_df.empty else [],
        store_id=store_id,
        item_id=item_id,
        llm_generator=llm_generator,
        api_client=api_client
    )
