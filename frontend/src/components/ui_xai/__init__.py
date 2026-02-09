"""
XAI (Explainable AI) Module
Provides SHAP-based explanations for sales forecasting model
All SHAP computations are now handled by Backend API
"""

# Plotting functions
from .shap_plots import (
    plot_global_feature_importance,
    plot_shap_beeswarm,
    plot_feature_importance_by_category,
    plot_shap_waterfall,
    plot_shap_dependence,
    create_local_explanation_table
)

# LLM Explainer
from .llm_explainer import (
    SalesInsightGenerator,
    create_insight_generator,
    FEATURE_DICTIONARY,
    CATEGORY_TRANSLATIONS
)

# Main view
from .xai_view import xai_explanation_view

__all__ = [
    # Plotting
    'plot_global_feature_importance',
    'plot_shap_beeswarm',
    'plot_feature_importance_by_category',
    'plot_shap_waterfall',
    'plot_shap_dependence',
    'create_local_explanation_table',
    # LLM
    'SalesInsightGenerator',
    'create_insight_generator',
    'FEATURE_DICTIONARY',
    'CATEGORY_TRANSLATIONS',
    # View
    'xai_explanation_view'
]
