"""
Local Explanations Section Component  
Displays instance-level (single prediction) explanations with waterfall plots
"""

import streamlit as st
import numpy as np
from ..shap_plots import plot_shap_waterfall, create_local_explanation_table
from .ai_button import render_ai_analysis_button


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
    
    sample_idx = st.slider(
        "Select a prediction to explain:",
        min_value=0,
        max_value=num_samples - 1,
        value=0,
        help=f"Choose from {num_samples} samples"
    )
    
    # Calculate metrics
    sample_date = df_sample.iloc[sample_idx]['date']
    sample_actual = np.expm1(df_sample.iloc[sample_idx]['logunits'])
    
    sample_prediction = expected_value + shap_values[sample_idx].sum()
    predicted_units = np.expm1(sample_prediction)
    
    error_diff = predicted_units - sample_actual
    error_pct = 100 * error_diff / sample_actual if sample_actual > 0 else 0
    
    # Display Metrics in aligned row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("📅 Date", sample_date.strftime('%Y-%m-%d'))
    with m2:
        st.metric("📉 Actual Sales", f"{sample_actual:.1f}")
    with m3:
        st.metric("📈 Model Prediction", f"{predicted_units:.1f}", delta=f"{error_diff:.1f}")
    with m4:
        st.metric("⚠️ Error %", f"{error_pct:.1f}%")
    
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
        st.subheader("Influencing Factors")
        
        increasing, decreasing = create_local_explanation_table(
            feature_names,
            X_sample.iloc[sample_idx].values,
            shap_values[sample_idx],
            top_n=5
        )
        
        st.markdown("🟢 *Increasing Prediction:*")
        st.dataframe(
            increasing.style.format({'Feature Value': '{:.2f}', 'SHAP Impact': '{:.4f}'}),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("🔴 *Decreasing Prediction:*")
        st.dataframe(
            decreasing.style.format({'Feature Value': '{:.2f}', 'SHAP Impact': '{:.4f}'}),
            hide_index=True,
            use_container_width=True
        )
    
    # AI Local Explanation Button
    if llm_generator is not None:
        render_ai_analysis_button(
            button_text="🤖 Explain This Prediction",
            button_key="local_explain_btn",
            llm_generator=llm_generator,
            fig=fig,
            generate_func=llm_generator.generate_local_explanation,
            title="🤖 Prediction Analysis",
            figure_prefix=f"waterfall_{sample_idx}",
            store_nbr=store_nbr,
            item_nbr=item_nbr,
            date=sample_date.strftime('%Y-%m-%d'),
            actual_value=sample_actual,
            predicted_value=predicted_units,
            increasing_factors=increasing,
            decreasing_factors=decreasing
        )
    else:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.info("💡 Enter API Key to enable AI explanations")
