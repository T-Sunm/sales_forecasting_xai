"""
Feature Dependence Analysis Section Component
Displays SHAP dependence plots for selected features
"""

import streamlit as st
from ..shap_plots import plot_shap_dependence
from .ai_button import render_ai_analysis_button


def display_dependence_analysis(
    shap_values,
    X_sample,
    importance_df,
    llm_generator=None,
    store_nbr=None,
    item_nbr=None
):
    """
    Display SHAP dependence plots for top features
    
    Args:
        shap_values: numpy array of SHAP values
        X_sample: Feature data DataFrame
        importance_df: Feature importance DataFrame
        llm_generator: SalesInsightGenerator instance
        store_nbr: Store ID for context
        item_nbr: Item ID for context
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
    
    # AI Feature Analysis Button
    if llm_generator is not None:
        render_ai_analysis_button(
            button_text="🤖 Explain Feature Impact",
            button_key="feature_analysis_btn",  
            llm_generator=llm_generator,
            fig=fig,
            generate_func=llm_generator.generate_feature_dependence_analysis,
            title=f"🤖 Feature Analysis: {selected_feature}",
            figure_prefix=f"dependence_{selected_feature}",
            feature_name=selected_feature,
            feature_stats=feature_stats,
            category=category,
            store_nbr=store_nbr,
            item_nbr=item_nbr
        )
    else:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.info("💡 Enter API Key to unlock feature insights")
