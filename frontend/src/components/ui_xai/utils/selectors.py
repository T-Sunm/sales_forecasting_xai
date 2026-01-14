"""
Selectors and UI components for XAI view
Store/Item selector and LLM generator initialization
"""

import os
import streamlit as st
from ..llm_explainer import SalesInsightGenerator


def create_store_item_selector(models_list):
    """
    Create sidebar selector for store and item
    
    Args:
        models_list: List of dicts [{'store_id': 1, 'item_id': 2}, ...] from API
    
    Returns:
        tuple: (store_nbr, item_nbr)
    """
    st.sidebar.header("🎯 Select Store & Item")
    
    if not models_list:
        st.sidebar.error("No models available")
        return None, None
    
    # Extract unique stores
    available_stores = sorted(list(set(m['store_id'] for m in models_list)))
    
    # Store selection
    store_nbr = st.sidebar.selectbox(
        "Store",
        available_stores,
        help="Select a store to analyze"
    )
    
    # Get available items for selected store
    available_items = sorted([
        m['item_id'] for m in models_list 
        if m['store_id'] == store_nbr
    ])
    
    if len(available_items) == 0:
        st.sidebar.warning(f"No items available for Store {store_nbr}")
        return store_nbr, None
    
    # Item selection
    item_nbr = st.sidebar.selectbox(
        "Item",
        available_items,
        help="Select an item to analyze"
    )
    
    return store_nbr, item_nbr


def init_llm_generator():
    """
    Initialize LLM generator with API key from sidebar or environment
    
    Returns:
        SalesInsightGenerator or None
    """
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI Insights")
    
    # Check for existing API key in environment
    env_key = os.environ.get("GROQ_API_KEY", "")
    
    # API key input
    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Enter your Groq API key to enable AI-generated insights"
    )
    
    if api_key:
        try:
            generator = SalesInsightGenerator(api_key=api_key)
            st.sidebar.success("✅ LLM Ready")
            return generator
        except Exception as e:
            st.sidebar.error(f"❌ LLM Error: {str(e)}")
            return None
    else:
        st.sidebar.warning("⚠️ Enter API key for AI insights")
        return None
