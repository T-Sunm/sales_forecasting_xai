"""
Feature Dependence Analysis Section Component
Displays SHAP dependence plots for selected features
"""

import streamlit as st
import pandas as pd
from ..shap_plots import plot_shap_dependence, plot_api_dependence
from .ai_button import render_ai_analysis_button


def display_dependence_analysis(
    shap_values,
    X_sample,
    importance_df,
    llm_generator=None,
    store_id=None,
    item_id=None,
    api_client=None
):
    """
    Display SHAP dependence plots for top features
    
    Args:
        shap_values: numpy array of SHAP values
        X_sample: Feature data DataFrame
        importance_df: Feature importance DataFrame
        llm_generator: SalesInsightGenerator instance
        store_id: Store ID for context
        item_id: Item ID for context
        api_client: APIClient instance (optional, for fetching data)
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
    
    # Plot and stats layout
    col1, col2 = st.columns([3, 1])
    
    fig = None
    feature_stats = {}
    category = "Unknown"
    
    if api_client:
        # Fetch data from API
        with st.spinner(f"Fetching dependence data for {selected_feature}..."):
            dep_data = api_client.get_dependence(store_id, item_id, selected_feature)
        
        if dep_data:
            with col1:
                fig = plot_api_dependence(dep_data)
                st.pyplot(fig)
            
            stats = dep_data.get('stats', {})
            feature_stats = {
                'mean': stats.get('mean', 0),
                'std': stats.get('std', 0),
                'min': stats.get('min', 0),
                'max': stats.get('max', 0)
            }
        else:
            st.error("Failed to load dependence data")
            return
    else:
        # Local plotting
        with col1:
            fig = plot_shap_dependence(
                shap_values,
                X_sample,
                selected_feature
            )
            st.pyplot(fig)
        
        desc = X_sample[selected_feature].describe()
        feature_stats = {
            'mean': desc['mean'],
            'std': desc['std'],
            'min': desc['min'],
            'max': desc['max']
        }
    
    category = importance_df[importance_df['feature'] == selected_feature]['category'].values[0] if not importance_df.empty else "Unknown"

    with col2:
        st.markdown(f"**About {selected_feature}:**")
        
        # Show statistics
        st.markdown(f"""
        - **Mean:** {feature_stats['mean']:.2f}
        - **Std:** {feature_stats['std']:.2f}
        - **Min:** {feature_stats['min']:.2f}
        - **Max:** {feature_stats['max']:.2f}
        """)
        
        # Feature category
        st.info(f"**Category:** {category}")
    
    # AI Feature Analysis Button
    if llm_generator is not None and fig:
        render_ai_analysis_button(
            button_text="🤖 Explain Feature Impact",
            button_key="feature_analysis_btn",  
            llm_generator=llm_generator,
            fig=fig,
            generate_func=llm_generator.generate_feature_dependence_analysis,
            title=f"🤖 Feature Analysis: {selected_feature}",
            figure_prefix=f"dependence_{selected_feature}",
            feature_name=selected_feature,
            feature_stats=pd.Series(feature_stats), # LLM expects Series
            category=category,
            store_id=store_id,
            item_id=item_id
        )
    elif llm_generator is None:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.info("💡 Enter API Key to unlock feature insights")
