# XAI Module — Explainable AI for Sales Forecasting

This module encapsulates the rendering logic and analytical interpretation of Explainable AI (XAI) metrics, specifically relying on SHAP (SHapley Additive exPlanations) values, to demystify the sales forecasting predictive models.

## 📁 Architectural Structure

```text
src/ui_xai/
├── __init__.py           # Module exports and access interfaces
├── explainer.py          # Core SHAP computation and feature categorization logic
├── shap_plots.py         # Matplotlib-based visualization rendering routines
└── xai_view.py           # Streamlit view composition and dashboard layout management
```

## 🚀 Analytical Capabilities

1. **Global Explanations:**
   - **Feature Importance:** Structural bar charts aggregating absolute SHAP values to expose macro-level feature impacts.
   - **SHAP Summary Plot:** Beeswarm coordinate distributions revealing granular feature value correlations relative to model output magnitudes.
   - **Category Analysis:** Heuristic grouping of attributes (e.g., Sales History, Weather conditions, Store Context) establishing definitive business-level interpretability.

2. **Dependency Analysis:**
   - Scatter plot matrices demonstrating isolated main effects and non-linear interactions mapping explicit feature variance to predictive deviations.

3. **Local Explanations:**
   - Waterfall visualizations delineating the exact additive SHAP vectors driving an individual atomic prediction relative to the baseline expected value.

---

## 🎯 Implementation Guide

### Streamlit Integration

Embed the interface securely within the root application context:

```python
from src.ui_xai import xai_explanation_view

# Delegate UI layout management to the module
xai_explanation_view(data, models_dict, feature_engineered_data)
```

### Programmatic Invocation

Isolate module functions for custom analytical executions:

```python
from src.ui_xai import (
    get_model_for_store_item,
    prepare_sample_data,
    compute_shap_values,
    plot_global_feature_importance
)

# 1. Isolate the target inference artifact
model = get_model_for_store_item(models_dict, store_nbr=37, item_nbr=105)

# 2. Extract a systematically randomized representative sample
X_sample, df_sample = prepare_sample_data(
    feature_engineered_data, store_nbr=37, item_nbr=105, feature_cols=feature_columns
)

# 3. Compute predictive attributions (SHAP values)
shap_values, expected_value = compute_shap_values(model, X_sample.values, feature_cols)

# 4. Generate standard visual artifacts
fig = plot_global_feature_importance(shap_values, feature_cols, top_n=20)
```

---

## ⚡ Performance Paradigms

1. **Memoization:** Computationally expensive TreeExplainer operations are persistently cached during runtime via Streamlit's native `@st.cache_data`.
2. **Deterministic Sampling:** Intensive computations explicitly subset vectors to a configured median (default `n=500` samples) forcing an optimal equilibrium between execution latency and statistical significance.
3. **Lazy Evaluation:** Component modules parameterize evaluation purely upon active component rendering.

## 📚 Core Dependencies

| Dependency | System Purpose |
| --- | --- |
| `shap` | Model interpretability and local explanations matrix generation |
| `streamlit` | Reactive UI framework mapping |
| `matplotlib` & `seaborn` | Statistical visualization layout execution |
| `numpy` & `pandas` | High-performance vectorized DataFrame operations |
