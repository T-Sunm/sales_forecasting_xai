"""
XAI SHAP Explainer Module
Helper functions for computing SHAP values and generating explanations
"""

import numpy as np
import pandas as pd
import shap
import math
from typing import Dict, List, Tuple, Any, Optional


def safe_float(val) -> float:
    """Convert value to JSON-safe float (replace NaN/Inf with 0)"""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except (ValueError, TypeError):
        return 0.0


def safe_float_list(arr) -> List[float]:
    """Convert numpy array to JSON-safe Python list"""
    return [safe_float(x) for x in arr]


def compute_shap_values(
    model,
    X_sample: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values for a model and sample data
    
    Args:
        model: Trained LightGBM model
        X_sample: Sample data (numpy array)
        feature_names: List of feature names
    
    Returns:
        tuple: (shap_values, expected_value)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
    
    # Clean NaN/Inf
    shap_values = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
    expected_value = safe_float(expected_value)
    
    return shap_values, expected_value


def get_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Calculate feature importance from SHAP values
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
    
    Returns:
        List of dicts with feature, importance, and category
    """
    # Replace NaN/Inf in SHAP values
    shap_values = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate mean absolute SHAP value for each feature
    importance = np.abs(shap_values).mean(axis=0)
    
    # Replace any remaining NaN/Inf
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create feature importance list
    feature_importance = []
    for idx, feat in enumerate(feature_names):
        category = categorize_feature(feat)
        feature_importance.append({
            "feature": feat,
            "importance": safe_float(importance[idx]),
            "category": category
        })
    
    # Sort by importance descending
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return feature_importance


def categorize_feature(feature_name: str) -> str:
    """
    Categorize feature into groups for better understanding
    
    Args:
        feature_name: Name of the feature
    
    Returns:
        Category name
    """
    feature_lower = feature_name.lower()
    
    # Sales history features
    if any(x in feature_lower for x in ["logunits", "lag", "mean", "max", "min", "std", "ewma"]):
        if "store_" in feature_lower:
            return "Store Context"
        elif "item_" in feature_lower:
            return "Item Context"
        else:
            return "Sales History"
    
    # Calendar features
    if any(x in feature_lower for x in ["day", "month", "year", "weekend", "holiday", "season", "blackfriday"]):
        return "Calendar & Events"
    
    # Weather features
    if any(x in feature_lower for x in ["tmax", "tmin", "precip", "pressure", "wind", "cool", "heat", "sealevel"]):
        return "Weather Conditions"
    
    # Weather codes
    if feature_name.isupper() and len(feature_name) <= 6:
        return "Weather Codes"
    
    return "Other"


def get_category_summary(
    feature_importance: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Summarize importance by category
    
    Args:
        feature_importance: List of feature importance dicts
    
    Returns:
        List of category summaries
    """
    categories = {}
    total_importance = sum(f["importance"] for f in feature_importance)
    
    for feat in feature_importance:
        cat = feat["category"]
        if cat not in categories:
            categories[cat] = {
                "category": cat,
                "total_importance": 0.0,
                "num_features": 0
            }
        categories[cat]["total_importance"] += feat["importance"]
        categories[cat]["num_features"] += 1
    
    # Calculate percentages
    summary = []
    for cat, data in categories.items():
        pct = (data["total_importance"] / total_importance * 100) if total_importance > 0 else 0
        summary.append({
            "category": cat,
            "importance_pct": round(pct, 2),
            "num_features": data["num_features"]
        })
    
    # Sort by percentage descending
    summary.sort(key=lambda x: x["importance_pct"], reverse=True)
    
    return summary


def get_dependence_data(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    feature_name: str,
    interaction_feature: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get SHAP dependence plot data for a feature
    
    Args:
        shap_values: SHAP values array
        X_sample: Sample data DataFrame
        feature_name: Target feature name
        interaction_feature: Optional interaction feature for coloring
    
    Returns:
        Dict with feature values, shap values, and interaction values
    """
    # Replace NaN/Inf in SHAP values
    shap_values = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    feature_idx = X_sample.columns.get_loc(feature_name)
    feature_shap = shap_values[:, feature_idx]
    feature_values = X_sample[feature_name].values
    
    # Replace NaN in feature values
    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    feature_shap = np.nan_to_num(feature_shap, nan=0.0, posinf=0.0, neginf=0.0)
    
    result = {
        "feature": feature_name,
        "feature_values": safe_float_list(feature_values),
        "shap_values": safe_float_list(feature_shap),
        "stats": {
            "mean": safe_float(np.mean(feature_values)),
            "std": safe_float(np.std(feature_values)),
            "min": safe_float(np.min(feature_values)),
            "max": safe_float(np.max(feature_values))
        }
    }
    
    # Add interaction feature if specified
    if interaction_feature and interaction_feature in X_sample.columns:
        interaction_values = X_sample[interaction_feature].values
        result["interaction_feature"] = interaction_feature
        result["interaction_values"] = safe_float_list(interaction_values)
    
    return result


def get_local_explanation(
    shap_values: np.ndarray,
    expected_value: float,
    X_sample: pd.DataFrame,
    sample_idx: int,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Get local SHAP explanation for a specific sample
    
    Args:
        shap_values: SHAP values array
        expected_value: Model's expected value (base value)
        X_sample: Sample data DataFrame
        sample_idx: Index of sample to explain
        top_n: Number of top features to return
    
    Returns:
        Dict with base value, prediction, and feature contributions
    """
    sample_shap = shap_values[sample_idx]
    sample_features = X_sample.iloc[sample_idx]
    
    # Replace NaN/Inf in SHAP values
    sample_shap = np.nan_to_num(sample_shap, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate prediction
    prediction = expected_value + np.sum(sample_shap)
    
    # Get feature contributions
    features = []
    for feat_name, shap_val, feat_val in zip(X_sample.columns, sample_shap, sample_features):
        # Handle NaN in feature values
        if pd.isna(feat_val):
            feat_val = 0.0
        features.append({
            "feature": feat_name,
            "value": safe_float(feat_val),
            "shap_impact": safe_float(shap_val)
        })
    
    # Sort by absolute SHAP value
    features.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
    
    # Split into increasing and decreasing
    increasing = [f for f in features if f["shap_impact"] > 0][:top_n]
    decreasing = [f for f in features if f["shap_impact"] < 0][:top_n]
    
    return {
        "base_value": safe_float(expected_value),
        "prediction": safe_float(prediction),
        "features": features[:top_n],
        "increasing_factors": increasing,
        "decreasing_factors": decreasing
    }
