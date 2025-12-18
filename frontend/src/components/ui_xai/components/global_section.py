"""
Global Explanations Section Component
Displays global feature importance with 3 tabs: Top Features, Categories, SHAP Summary
"""

import streamlit as st
from ..shap_plots import (
    plot_global_feature_importance,
    plot_feature_importance_by_category,
    plot_shap_beeswarm
)
from ..utils import XAIStateManager, save_figure_to_temp, cleanup_temp_file
from .ai_button import render_ai_analysis_button


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
    
    # Render each tab
    _render_tab1_top_features(
        shap_values, feature_names, importance_df, category_summary,
        store_nbr, item_nbr, llm_generator, tab1
    )
    
    _render_tab2_categories(
        importance_df, category_summary,
        store_nbr, item_nbr, llm_generator, tab2
    )
    
    _render_tab3_beeswarm(
        shap_values, X_sample, importance_df, category_summary,
        store_nbr, item_nbr, llm_generator, tab3
    )


def _render_tab1_top_features(
    shap_values, feature_names, importance_df, category_summary,
    store_nbr, item_nbr, llm_generator, tab
):
    """Render Tab 1: Top Features"""
    with tab:
        st.subheader("Top 20 Most Important Features")
        
        # Plot and table layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_top_features = plot_global_feature_importance(
                shap_values,
                feature_names,
                top_n=20,
                importance_df=importance_df
            )
            if fig_top_features:
                st.pyplot(fig_top_features)
            else:
                st.warning("⚠️ Cannot plot top features")
        
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
        
        # AI Analysis Button
        if fig_top_features:
            render_ai_analysis_button(
                button_text="✨ Analyze Top Features",
                button_key="tab1_ai_btn",
                llm_generator=llm_generator,
                fig=fig_top_features,
                generate_func=llm_generator.generate_global_report if llm_generator else None,
                title="🤖 Top Features Analysis",
                figure_prefix="top_features",
                store_nbr=store_nbr,
                item_nbr=item_nbr,
                importance_df=importance_df,
                category_summary=category_summary,
                tab_type="top_features"
            )


def _render_tab2_categories(
    importance_df, category_summary,
    store_nbr, item_nbr, llm_generator, tab
):
    """Render Tab 2: Feature Categories"""
    with tab:
        st.subheader("Feature Importance by Category")
        st.markdown("""
        Features are grouped into business categories to show which 
        **types of information** drive predictions.
        """)
        
        # Category breakdown
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig_categories = plot_feature_importance_by_category(
                importance_df,
                category_summary,
                figsize=(14, 8)
            )
            st.pyplot(fig_categories)
        
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
        
        # AI Analysis Button
        render_ai_analysis_button(
            button_text="✨ Analyze Categories",
            button_key="tab2_ai_btn",
            llm_generator=llm_generator,
            fig=fig_categories,
            generate_func=llm_generator.generate_global_report if llm_generator else None,
            title="🤖 Category Analysis",
            figure_prefix="categories",
            store_nbr=store_nbr,
            item_nbr=item_nbr,
            importance_df=importance_df,
            category_summary=category_summary,
            tab_type="categories"
        )


def _render_tab3_beeswarm(
    shap_values, X_sample, importance_df, category_summary,
    store_nbr, item_nbr, llm_generator, tab
):
    """Render Tab 3: SHAP Summary (Beeswarm) Plot"""
    with tab:
        st.subheader("SHAP Summary (Beeswarm) Plot")
        st.markdown("""
        Each dot is a data point. Color shows feature value (red=high, blue=low).
        Position shows SHAP impact on prediction.
        """)
        
        if shap_values is None or X_sample is None:
            st.info("ℹ️ SHAP Beeswarm plot requires local raw data processing. Currently using API mode.")
            st.warning("Please switch to local mode or check back later for detailed SHAP plots.")
            return

        # Generate the plot
        fig_beeswarm = plot_shap_beeswarm(
            shap_values,
            X_sample,
            max_display=15
        )
        
        # Initialize session state
        XAIStateManager.init_tab_state(XAIStateManager.TAB3_ANALYSIS, False)
        
        # Dynamic layout based on state
        if not XAIStateManager.get(XAIStateManager.TAB3_ANALYSIS, False):
            # Initial state: Centered small plot
            col1, col2, col3 = st.columns([0.5, 1, 0.5])
            with col2:
                st.pyplot(fig_beeswarm)
        else:
            # Expanded state: 2-column with analysis
            col_plot, col_analysis = st.columns([1.2, 1])
            
            with col_plot:
                st.pyplot(fig_beeswarm)
            
            with col_analysis:
                with st.spinner("AI đang phân tích (Vision Model)..."):
                    try:
                        image_path = save_figure_to_temp(fig_beeswarm, "beeswarm")
                        
                        report = llm_generator.generate_global_report(
                            store_nbr=store_nbr,
                            item_nbr=item_nbr,
                            importance_df=importance_df,
                            category_summary=category_summary,
                            image_path=image_path,
                            tab_type="beeswarm"
                        )
                        
                        st.markdown("### 🤖 Pattern Analysis")
                        st.markdown(report)
                        
                        cleanup_temp_file(image_path)
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")
        
        # Centered button (shown in both states)
        if llm_generator is not None:
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                btn_clicked = st.button("✨ Analyze SHAP Patterns", key="tab3_ai_btn", use_container_width=True)
                if btn_clicked and not XAIStateManager.get(XAIStateManager.TAB3_ANALYSIS, False):
                    XAIStateManager.set(XAIStateManager.TAB3_ANALYSIS, True)
                    st.rerun()
        else:
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                st.info("💡 Enter API Key to unlock AI insights")
