"""
SHAP Explainer Module
Handles SHAP value computation, caching, and feature classification
"""

import numpy as np
import pandas as pd
import streamlit as st
import shap


@st.cache_data
def get_model_for_store_item(_models_dict, store_nbr, item_nbr):
    """
    Lấy model tương ứng với cặp (store_nbr, item_nbr)
    
    Args:
        models_dict: Dictionary chứa tất cả models {(store, item): model}
        store_nbr: Store ID
        item_nbr: Item ID
    
    Returns:
        model: LightGBM model hoặc None nếu không tìm thấy
    """
    key = (store_nbr, item_nbr)
    
    if key in _models_dict:
        return _models_dict[key]
    
    # Try with different types (int vs float)
    for k in _models_dict.keys():
        if int(k[0]) == int(store_nbr) and int(k[1]) == int(item_nbr):
            return _models_dict[k]
    
    return None


def prepare_sample_data(
    feature_engineered_data,
    store_nbr,
    item_nbr,
    feature_cols,
    sample_size=500,
    random_state=42
):
    """
    Chuẩn bị data sample cho SHAP analysis
    
    Args:
        feature_engineered_data: Full feature DataFrame
        store_nbr: Store ID
        item_nbr: Item ID
        feature_cols: List of feature columns
        sample_size: Số lượng samples (default 500)
        random_state: Random seed
    
    Returns:
        X_sample: Sampled feature data (DataFrame)
        df_sample: Original data với metadata (DataFrame)
    """
    # Filter data for specific store-item pair
    df_filtered = feature_engineered_data[
        (feature_engineered_data['store_nbr'] == store_nbr) &
        (feature_engineered_data['item_nbr'] == item_nbr) &
        (feature_engineered_data['is_kaggle_test'] == 0)  # Only use train data
    ].copy()
    
    if len(df_filtered) == 0:
        st.error(f"No data found for store {store_nbr}, item {item_nbr}")
        return None, None
    
    # Sample data
    actual_sample_size = min(sample_size, len(df_filtered))
    df_sample = df_filtered.sample(n=actual_sample_size, random_state=random_state)
    
    # Extract features
    X_sample = df_sample[feature_cols].copy()
    
    return X_sample, df_sample


@st.cache_data(show_spinner=False)
def compute_shap_values(_model, X_sample_values, feature_names):
    """
    Tính SHAP values cho một tập data sample
    
    Args:
        _model: Trained LightGBM model (underscore để tránh hash bởi st.cache)
        X_sample_values: numpy array of feature values
        feature_names: List of feature names
    
    Returns:
        shap_values: numpy array SHAP values
        expected_value: Base value (expected value)
    """
    with st.spinner('Computing SHAP values...'):
        # Create SHAP explainer
        explainer = shap.TreeExplainer(_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample_values)
        
        # Get expected value (base value)
        expected_value = explainer.expected_value
        
    return shap_values, expected_value


def classify_feature(feature_name):
    """
    Phân loại features vào các nhóm business categories
    
    Feature Groups:
    1. Sales History - Autoregressive features
    2. Store Context - Store-level performance
    3. Item Context - Product-level features
    4. Calendar & Events - Time-based features
    5. Weather Conditions - Weather measurements
    6. Weather Codes - Weather phenomenon codes
    7. Other - Miscellaneous
    
    Args:
        feature_name: Tên feature (string)
    
    Returns:
        category: String tên category
    """
    feature = feature_name.lower()
    
    # 1. Sales History (Autoregressive features)
    if any(keyword in feature for keyword in [
        'logunits_lag', 'logunits_mean', 'logunits_min', 'logunits_max',
        'logunits_std', 'logunits_sum', 'logunits_ewma',
        'sales_lag', 'sales_mean', 'sales_std'
    ]):
        return 'Sales History'
    
    # 2. Store Context
    if 'store_' in feature and feature_name not in ['store_nbr']:
        return 'Store Context'
    
    # 3. Item/Product Context
    if 'item_' in feature and feature_name not in ['item_nbr']:
        return 'Item Context'
    
    # 4. Calendar/Time features
    if any(keyword in feature for keyword in [
        'day_of_week', 'month', 'year', 'day', 'is_weekend',
        'is_holiday', 'is_blackfriday', 'season_'
    ]):
        return 'Calendar & Events'
    
    # 5. Weather features (excluding weather code columns)
    if any(keyword in feature for keyword in [
        'tmax', 'tmin', 'temp', 'depart', 'cool', 'humidity',
        'precip', 'snowfall', 'pressure', 'sealevel',
        'sunrise', 'sunset', 'resultspeed', 'resultdir'
    ]):
        return 'Weather Conditions'
    
    # 6. Weather phenomenon codes (BCFG, BR, RA, etc.)
    # These are typically short uppercase codes
    if len(feature_name) <= 5 and feature_name.isupper():
        return 'Weather Codes'
    
    # 7. Everything else
    return 'Other'


def get_feature_importance_df(shap_values, feature_names):
    """
    Tạo DataFrame chứa feature importance từ SHAP values
    
    Args:
        shap_values: numpy array of SHAP values
        feature_names: List of feature names
    
    Returns:
        DataFrame với columns: feature, importance, category
    """
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    })
    
    # Add category classification
    importance_df['category'] = importance_df['feature'].apply(classify_feature)
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def get_category_summary(importance_df):
    """
    Tính tổng importance theo category
    
    Args:
        importance_df: DataFrame từ get_feature_importance_df()
    
    Returns:
        DataFrame với columns: category, total_importance, num_features, importance_pct
    """
    category_summary = (
        importance_df.groupby('category')
        .agg({
            'importance': 'sum',
            'feature': 'count'
        })
        .reset_index()
        .rename(columns={'feature': 'num_features'})
    )
    
    # Calculate percentage
    total_importance = category_summary['importance'].sum()
    category_summary['importance_pct'] = (
        100 * category_summary['importance'] / total_importance
    )
    
    # Sort by importance
    category_summary = category_summary.sort_values('importance', ascending=False)
    
    return category_summary


def get_top_features_per_category(importance_df, top_n=5):
    """
    Lấy top N features quan trọng nhất cho mỗi category
    
    Args:
        importance_df: DataFrame từ get_feature_importance_df()
        top_n: Số lượng top features per category
    
    Returns:
        DataFrame chứa top features per category
    """
    result = []
    
    for category in importance_df['category'].unique():
        category_df = importance_df[importance_df['category'] == category]
        top_features = category_df.nlargest(top_n, 'importance')
        result.append(top_features)
    
    return pd.concat(result, ignore_index=True)



