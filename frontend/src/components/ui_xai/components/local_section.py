"""
Local Explanations Section Component  
Displays instance-level (single prediction) explanations with waterfall plots
"""

import streamlit as st
import numpy as np
import pandas as pd
from ..shap_plots import plot_shap_waterfall, plot_api_waterfall, create_local_explanation_table
from .ai_button import render_ai_analysis_button


def display_local_explanations(
    shap_values,
    expected_value,
    X_sample,
    df_sample,
    feature_names,
    store_id=None,
    item_id=None,
    llm_generator=None,
    api_client=None
):
    """
    Display local (instance-level) explanations
    
    Args:
        shap_values: numpy array of SHAP values
        expected_value: Base value (expected output)
        X_sample: Feature data DataFrame
        df_sample: Original data with metadata
        feature_names: List of feature names
        store_id: Store ID for LLM context
        item_id: Item ID for LLM context
        llm_generator: SalesInsightGenerator instance
        api_client: APIClient instance (optional)
    """
    st.header("🎯 Local Prediction Explanations")
    st.markdown("""
    Understand **individual predictions** - why the model predicted a specific value
    for a particular date.
    """)
    
    fig = None
    increasing = pd.DataFrame()
    decreasing = pd.DataFrame()
    sample_date_str = "Unknown"
    predicted_units = 0.0
    actual_value = 0.0
    
    if api_client:
        # API Mode
        st.info("Using Backend API for SHAP explanations")
        
        # Simple selector for API mode
        explanation_type = st.radio(
            "Select Explanation Target:",
            ["Most Recent Prediction", "Specific Date"],
            horizontal=True
        )
        
        target_date = None
        if explanation_type == "Specific Date":
             target_date = st.date_input("Select Date").strftime("%Y-%m-%d")
        
        if st.button("🔍 Explain Prediction"):
            with st.spinner("Fetching explanation..."):
                explanation = api_client.get_local_explanation(
                    store_id, item_id, 
                    date=target_date, 
                    top_n=10
                )
            
            if explanation and not explanation.get("error"):
                # Parse response
                base_value = explanation.get('base_value', 0)
                prediction = explanation.get('prediction', 0)
                predicted_units = np.expm1(prediction)
                actual_log = explanation.get('actual')
                actual_value = np.expm1(actual_log) if actual_log is not None else 0.0
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("📈 Prediction (Units)", f"{predicted_units:.1f}")
                with m2:
                    st.metric("📊 Base Value", f"{base_value:.2f}")
                if actual_log is not None:
                    with m3:
                        st.metric("✅ Actual Sales", f"{actual_value:.1f}", delta=f"{predicted_units-actual_value:.1f}")
                
                # Layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("SHAP Waterfall Plot")
                    fig = plot_api_waterfall(explanation)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Influencing Factors")
                    
                    # Construct tables from 'increasing_factors' and 'decreasing_factors' 
                    # returned by API
                    inc_data = explanation.get('increasing_factors', [])
                    dec_data = explanation.get('decreasing_factors', [])
                    
                    increasing = pd.DataFrame(inc_data).rename(columns={
                        'feature': 'Feature', 'value': 'Feature Value', 'shap_impact': 'SHAP Impact'
                    })
                    decreasing = pd.DataFrame(dec_data).rename(columns={
                        'feature': 'Feature', 'value': 'Feature Value', 'shap_impact': 'SHAP Impact'
                    })
                    
                    st.markdown("🟢 *Increasing Prediction:*")
                    if not increasing.empty:
                        st.dataframe(
                            increasing.style.format({'Feature Value': '{:.2f}', 'SHAP Impact': '{:.4f}'}),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    st.markdown("🔴 *Decreasing Prediction:*")
                    if not decreasing.empty:
                        st.dataframe(
                            decreasing.style.format({'Feature Value': '{:.2f}', 'SHAP Impact': '{:.4f}'}),
                            hide_index=True,
                            use_container_width=True
                        )
                sample_date_str = target_date if target_date else "Most Recent"
                
            else:
                st.error("Failed to fetch explanation from API. Backend may have encountered an error.")
        
    else:
        # Local Mode
        if df_sample is None:
            st.error("Missing local data")
            return
            
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
        sample_date_str = sample_date.strftime('%Y-%m-%d')
        actual_value = np.expm1(df_sample.iloc[sample_idx]['logunits'])
        
        sample_prediction = expected_value + shap_values[sample_idx].sum()
        predicted_units = np.expm1(sample_prediction)
        
        error_diff = predicted_units - actual_value
        error_pct = 100 * error_diff / actual_value if actual_value > 0 else 0
        
        # Display Metrics in aligned row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("📅 Date", sample_date_str)
        with m2:
            st.metric("📉 Actual Sales", f"{actual_value:.1f}")
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
    if llm_generator is not None and fig:
        render_ai_analysis_button(
            button_text="🤖 Explain This Prediction",
            button_key="local_explain_btn",
            llm_generator=llm_generator,
            fig=fig,
            generate_func=llm_generator.generate_local_explanation,
            title="🤖 Prediction Analysis",
            figure_prefix="waterfall_explanation",
            store_id=store_id,
            item_id=item_id,
            date=sample_date_str,
            actual_value=actual_value,
            predicted_value=predicted_units,
            increasing_factors=increasing,
            decreasing_factors=decreasing
        )
    elif llm_generator is None:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.info("💡 Enter API Key to enable AI explanations")
