"""
AI Analysis Button Component
Reusable component for AI-powered analysis buttons with consistent layout and behavior
"""

import streamlit as st
from typing import Callable, Optional, Any, Dict
from ..utils.figure_utils import save_figure_to_temp, cleanup_temp_file


def render_ai_analysis_button(
    button_text: str,
    button_key: str,
    llm_generator,
    fig,
    generate_func: Callable,
    title: str = "🤖 Analysis",
    figure_prefix: str = "plot",
    spinner_text: str = "AI đang phân tích (Vision Model)...",
    **llm_kwargs
):
    """
    Render a centered AI analysis button with consistent behavior.
    
    This component handles:
    - Centered button layout (columns [1, 1.5, 1])
    - Figure saving to temp file
    - LLM API call with spinner
    - Result display with markdown
    - Cleanup of temp files
    - Error handling
    
    Args:
        button_text: Text to display on button (e.g., "✨ Analyze Top Features")
        button_key: Unique key for the button
        llm_generator: SalesInsightGenerator instance
        fig: Matplotlib figure to save and analyze
        generate_func: LLM generation function to call (e.g., llm_generator.generate_global_report)
        title: Title to display above analysis result
        figure_prefix: Prefix for temp file naming
        spinner_text: Text to show in spinner
        **llm_kwargs: Additional kwargs to pass to generate_func
    
    Returns:
        None (renders in place)
    
    Example:
        ```python
        render_ai_analysis_button(
            button_text="✨ Analyze Top Features",
            button_key="tab1_ai_btn",
            llm_generator=llm_generator,
            fig=fig_top_features,
            generate_func=llm_generator.generate_global_report,
            title="🤖 Top Features Analysis",
            figure_prefix="top_features",
            store_id=store_id,
            item_id=item_id,
            importance_df=importance_df,
            category_summary=category_summary,
            tab_type="top_features"
        )
        ```
    """
    # Check if LLM is available
    if llm_generator is None:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.info("💡 Enter API Key to unlock AI insights")
        return
    
    # Create centered button layout
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        # Render button
        if st.button(button_text, key=button_key, use_container_width=True):
            # Button clicked - run analysis
            with st.spinner(spinner_text):
                image_path = None
                try:
                    # Save figure to temp file
                    image_path = save_figure_to_temp(fig, figure_prefix)
                    
                    # Add image_path to kwargs
                    llm_kwargs['image_path'] = image_path
                    
                    # Call LLM generation function
                    report = generate_func(**llm_kwargs)
                    
                    # Display result
                    st.markdown(f"### {title}")
                    st.markdown(report)
                    
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
                
                finally:
                    # Always cleanup temp file
                    if image_path:
                        cleanup_temp_file(image_path)
