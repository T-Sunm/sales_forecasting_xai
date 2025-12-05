"""
XAI View Module
Main Streamlit view for Explainable AI dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from .explainer import (
    get_model_for_store_item,
    prepare_sample_data,
    compute_shap_values,
    get_feature_importance_df,
    get_category_summary,
    get_top_features_per_category
)

from .shap_plots import (
    plot_global_feature_importance,
    plot_shap_beeswarm,
    plot_feature_importance_by_category,
    plot_shap_waterfall,
    plot_shap_dependence,
    create_local_explanation_table
)

from .llm_explainer import SalesInsightGenerator, create_insight_generator


def xai_explanation_view(data, models_dict, feature_engineered_data):
    """
    Main XAI view displaying SHAP-based explanations
    
    Args:
        data: Preprocessed sales data
        models_dict: Dictionary of trained models {(store, item): model}
        feature_engineered_data: Feature engineered dataset
    """
    st.title("🔍 Explainable AI - Model Interpretability")
    st.markdown("""
    Understand **why** the model makes certain predictions using SHAP 
    (SHapley Additive exPlanations) analysis.
    """)
    
    # Get feature columns (exclude metadata)
    drop_cols = [
        'date', 'units', 'logunits',
        'is_kaggle_test', 'is_valid',
        'store_nbr', 'item_nbr', 'station_nbr', 
        'depart', 'sunrise', 'sunset', 'snowfall'
    ]
    feature_cols = [c for c in feature_engineered_data.columns if c not in drop_cols]
    
    # Sidebar: Store & Item selection
    store_nbr, item_nbr = create_store_item_selector(feature_engineered_data, models_dict)
    
    if store_nbr is None or item_nbr is None:
        st.warning("Please select a valid store and item combination.")
        return
    
    # Get model
    model = get_model_for_store_item(models_dict, store_nbr, item_nbr)
    
    if model is None:
        st.error(f"No model found for Store {store_nbr}, Item {item_nbr}")
        return
    
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
    
    st.success(f"✅ Loaded {len(X_sample)} samples for Store {store_nbr}, Item {item_nbr}")
    
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
        importance_df
    )
    
    st.markdown("---")
    
    # Section 3: Local Explanations
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


def create_store_item_selector(feature_engineered_data, models_dict):
    """
    Create sidebar selector for store and item
    
    Args:
        feature_engineered_data: Feature dataset
        models_dict: Dictionary of models
    
    Returns:
        tuple: (store_nbr, item_nbr)
    """
    st.sidebar.header("🎯 Select Store & Item")
    
    # Get available (store, item) pairs from models_dict
    available_pairs = list(models_dict.keys())
    
    if len(available_pairs) == 0:
        st.sidebar.error("No models available")
        return None, None
    
    # Extract unique stores and items
    available_stores = sorted(set(int(pair[0]) for pair in available_pairs))
    
    # Store selection
    store_nbr = st.sidebar.selectbox(
        "Store",
        available_stores,
        help="Select a store to analyze"
    )
    
    # Get available items for selected store
    available_items = sorted([
        int(pair[1]) for pair in available_pairs 
        if int(pair[0]) == store_nbr
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
    
    # Show info
    st.sidebar.info(f"""
    **Selected:**
    - Store: {store_nbr}
    - Item: {item_nbr}
    """)
    
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


def display_global_explanations(
    shap_values,
    X_sample,
    feature_names,
    importance_df,
    category_summary,
    store_nbr=None,
    item_nbr=None,
    llm_generator=None
):
    """
    Display global feature importance and category analysis
    
    Args:
        shap_values: numpy array of SHAP values
        X_sample: Feature data DataFrame
        feature_names: List of feature names
        importance_df: Feature importance DataFrame
        category_summary: Category summary DataFrame
        store_nbr: Store ID for LLM context
        item_nbr: Item ID for LLM context
        llm_generator: SalesInsightGenerator instance
    """
    st.header("📊 Global Feature Importance")
    st.markdown("""
    These visualizations show which features are most important **across all predictions**.
    """)
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs([
        "Top Features",
        "Feature Categories",
        "SHAP Summary Plot"
    ])
    
    with tab1:
        st.subheader("Top 20 Most Important Features")
        
        # Show table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_global_feature_importance(
                shap_values,
                feature_names,
                top_n=20
            )
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Top 10 Features:**")
            top_10 = importance_df.head(10)[['feature', 'importance', 'category']]
            top_10_display = top_10.copy()
            top_10_display['importance'] = top_10_display['importance'].apply(lambda x: f"{x:.6f}")
            st.dataframe(
                top_10_display,
                hide_index=True,
                use_container_width=True
            )
    
    with tab2:
        st.subheader("Feature Importance by Category")
        st.markdown("""
        Features are grouped into business categories to show which 
        **types of information** drive predictions.
        """)
        
        # Category breakdown
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig = plot_feature_importance_by_category(
                importance_df,
                category_summary,
                figsize=(14, 8)
            )
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Category Summary:**")
            summary_display = category_summary.copy()
            summary_display['importance_pct'] = summary_display['importance_pct'].apply(
                lambda x: f"{x:.1f}%"
            )
            st.dataframe(
                summary_display[['category', 'num_features', 'importance_pct']],
                hide_index=True,
                use_container_width=True
            )
    
    with tab3:
        st.subheader("SHAP Summary (Beeswarm) Plot")
        st.markdown("""
        Each dot is a data point. Color shows feature value (red=high, blue=low).
        Position shows SHAP impact on prediction.
        """)
        
        fig = plot_shap_beeswarm(
            shap_values,
            X_sample,
            max_display=15
        )
        st.pyplot(fig)
    
    # AI Generated Report Section
    st.markdown("---")
    st.subheader("🤖 AI-Generated Analysis")
    
    if llm_generator is not None:
        if st.button("Generate Global Insights Report", key="global_report_btn"):
            with st.spinner("AI đang phân tích..."):
                try:
                    report = llm_generator.generate_global_report(
                        store_nbr=store_nbr,
                        item_nbr=item_nbr,
                        importance_df=importance_df,
                        category_summary=category_summary
                    )
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Lỗi khi tạo báo cáo: {str(e)}")
    else:
        st.info("💡 Nhập Groq API Key ở sidebar để sử dụng AI Insights")


def display_dependence_analysis(shap_values, X_sample, importance_df):
    """
    Display SHAP dependence plots for top features
    
    Args:
        shap_values: numpy array of SHAP values
        X_sample: Feature data DataFrame
        importance_df: Feature importance DataFrame
    """
    st.header("🔗 Feature Dependence Analysis")
    st.markdown("""
    Explore how feature **values** relate to their **SHAP impact** on predictions.
    """)
    
    # Get top features
    top_features = importance_df.head(10)['feature'].tolist()
    
    # Feature selector
    selected_feature = st.selectbox(
        "Select a feature to analyze:",
        top_features,
        help="Choose from top 10 most important features"
    )
    
    # Plot
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = plot_shap_dependence(
            shap_values,
            X_sample,
            selected_feature
        )
        st.pyplot(fig)
    
    with col2:
        st.markdown(f"**About {selected_feature}:**")
        
        # Show statistics
        feature_stats = X_sample[selected_feature].describe()
        st.markdown(f"""
        - **Mean:** {feature_stats['mean']:.2f}
        - **Std:** {feature_stats['std']:.2f}
        - **Min:** {feature_stats['min']:.2f}
        - **Max:** {feature_stats['max']:.2f}
        """)
        
        # Feature category
        category = importance_df[importance_df['feature'] == selected_feature]['category'].values[0]
        st.info(f"**Category:** {category}")


def display_local_explanations(
    shap_values,
    expected_value,
    X_sample,
    df_sample,
    feature_names,
    store_nbr=None,
    item_nbr=None,
    llm_generator=None
):
    """
    Display local (instance-level) explanations
    
    Args:
        shap_values: numpy array of SHAP values
        expected_value: Base value (expected output)
        X_sample: Feature data DataFrame
        df_sample: Original data with metadata
        feature_names: List of feature names
        store_nbr: Store ID for LLM context
        item_nbr: Item ID for LLM context
        llm_generator: SalesInsightGenerator instance
    """
    st.header("🎯 Local Prediction Explanations")
    st.markdown("""
    Understand **individual predictions** - why the model predicted a specific value
    for a particular date.
    """)
    
    # Sample selector
    num_samples = len(df_sample)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sample_idx = st.slider(
            "Select a prediction to explain:",
            min_value=0,
            max_value=num_samples - 1,
            value=0,
            help=f"Choose from {num_samples} samples"
        )
    
    with col2:
        # Show sample info
        sample_date = df_sample.iloc[sample_idx]['date']
        sample_actual = np.expm1(df_sample.iloc[sample_idx]['logunits'])
        
        st.metric("Date", sample_date.strftime('%Y-%m-%d'))
        st.metric("Actual Sales", f"{sample_actual:.1f} units")
    
    # Get prediction
    sample_prediction = expected_value + shap_values[sample_idx].sum()
    predicted_units = np.expm1(sample_prediction)
    
    st.markdown("---")
    
    # Layout: Waterfall + Tables
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("SHAP Waterfall Plot")
        st.markdown(f"""
        Shows how each feature **pushed** the prediction from the base value 
        ({expected_value:.2f}) to the final prediction ({sample_prediction:.2f}).
        """)
        
        fig = plot_shap_waterfall(
            shap_values,
            expected_value,
            X_sample.values,
            feature_names,
            sample_idx=sample_idx,
            max_display=12
        )
        st.pyplot(fig)
    
    with col2:
        st.subheader("Prediction Summary")
        
        # Metrics
        st.metric(
            "Model Prediction",
            f"{predicted_units:.1f} units",
            delta=f"{predicted_units - sample_actual:.1f} vs actual"
        )
        
        error_pct = 100 * (predicted_units - sample_actual) / sample_actual if sample_actual > 0 else 0
        st.metric("Error %", f"{error_pct:.1f}%")
        
        # Top factors
        st.markdown("---")
        st.markdown("**Top Influencing Factors:**")
        
        increasing, decreasing = create_local_explanation_table(
            feature_names,
            X_sample.iloc[sample_idx].values,
            shap_values[sample_idx],
            top_n=5
        )
        
        st.markdown("*Increasing Prediction:*")
        st.dataframe(
            increasing.style.format({'Feature Value': '{:.2f}', 'SHAP Impact': '{:.4f}'}),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("*Decreasing Prediction:*")
        st.dataframe(
            decreasing.style.format({'Feature Value': '{:.2f}', 'SHAP Impact': '{:.4f}'}),
            hide_index=True,
            use_container_width=True
        )
        
        # AI Local Explanation
        st.markdown("---")
        if llm_generator is not None:
            if st.button("🤖 Explain This Prediction", key="local_explain_btn"):
                with st.spinner("AI đang giải thích..."):
                    try:
                        explanation = llm_generator.generate_local_explanation(
                            store_nbr=store_nbr,
                            item_nbr=item_nbr,
                            date=sample_date.strftime('%Y-%m-%d'),
                            actual_value=sample_actual,
                            predicted_value=predicted_units,
                            increasing_factors=increasing,
                            decreasing_factors=decreasing
                        )
                        st.markdown("**🤖 AI Explanation:**")
                        st.markdown(explanation)
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")
        else:
            st.info("💡 Nhập API Key để AI giải thích dự báo này")
