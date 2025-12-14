"""
XAI View Module
Main Streamlit view for Explainable AI dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np

from .explainer import (
    get_model_for_store_item,
    prepare_sample_data,
    compute_shap_values,
    get_feature_importance_df,
    get_category_summary
)

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


def xai_explanation_view(data, models_dict, feature_engineered_data):
    """
    Main XAI view displaying SHAP-based explanations
    
    Args:
        data: Preprocessed sales data
        models_dict: Dictionary of trained models {(store, item): model}
        feature_engineered_data: Feature engineered dataset
    """
    st.title("🔍 Explainable AI (XAI) Dashboard")
    st.markdown("""
    Understand **how** and **why** the model makes predictions using SHAP 
    (SHapley Additive exPlanations) analysis.
    """)
    
    # Sidebar: Store and Item selection
    store_nbr, item_nbr = create_store_item_selector(
        feature_engineered_data,
        models_dict
    )
    
    if store_nbr is None or item_nbr is None:
        st.warning("⚠️ Please select both Store and Item from sidebar")
        return
    

    
    # Load model
    model = get_model_for_store_item(
        models_dict,
        store_nbr,
        item_nbr
    )
    
    if model is None:
        st.error(f"No model found for Store {store_nbr}, Item {item_nbr}")
        return
    
    # Get feature names from model
    feature_cols = model.feature_name_
    
    # Prepare sample data
    with st.spinner('Preparing data...'):
        X_sample, df_sample = prepare_sample_data(
            feature_engineered_data,
            store_nbr,
            item_nbr,
            feature_cols,
            sample_size=500
        )
    
    if X_sample is None:
        return
    

    
    # Compute SHAP values
    shap_values, expected_value = compute_shap_values(
        model,
        X_sample.values,
        feature_cols
    )
    
    # Get importance DataFrames
    importance_df = get_feature_importance_df(shap_values, feature_cols)
    category_summary = get_category_summary(importance_df)
    
    # Initialize LLM (if API key available)
    llm_generator = init_llm_generator()
    
    # Reset state if selection changed
    XAIStateManager.reset_on_selection_change(store_nbr, item_nbr)
    
    # Display sections
    st.markdown("---")
    
    # Section 1: Global Explanations
    display_global_explanations(
        shap_values,
        X_sample,
        feature_cols,
        importance_df,
        category_summary,
        store_nbr,
        item_nbr,
        llm_generator
    )
    
    st.markdown("---")
    
    # Section 2: Feature Dependency Analysis
    display_dependence_analysis(
        shap_values,
        X_sample,
        importance_df,
        llm_generator,
        store_nbr,
        item_nbr
    )
    
    st.markdown("---")
    
    # Section 3: Local (Instance) Explanations
    display_local_explanations(
        shap_values,
        expected_value,
        X_sample,
        df_sample,
        feature_cols,
        store_nbr,
        item_nbr,
        llm_generator
    )
