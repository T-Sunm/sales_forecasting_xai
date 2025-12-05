"""
SHAP Visualization Module
Provides clean, reusable plotting functions for SHAP analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch


# Set default plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')


def plot_global_feature_importance(
    shap_values,
    feature_names,
    top_n=20,
    figsize=(10, 8),
    title='Global Feature Importance'
):
    """
    Plot global feature importance as horizontal bar chart
    
    Args:
        shap_values: numpy array of SHAP values (n_samples, n_features)
        feature_names: list of feature names
        top_n: number of top features to display
        figsize: tuple of (width, height)
        title: plot title
    
    Returns:
        matplotlib.figure.Figure object
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bars
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'], color='steelblue', alpha=0.8)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_shap_beeswarm(
    shap_values,
    X_sample,
    max_display=20,
    figsize=(12, 8)
):
    """
    Create SHAP beeswarm (summary) plot
    
    Args:
        shap_values: numpy array of SHAP values
        X_sample: DataFrame of feature values
        max_display: maximum number of features to display
        figsize: tuple of (width, height)
    
    Returns:
        matplotlib.figure.Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create beeswarm plot
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
        plot_type='dot'
    )
    
    # Get current axis and customize
    ax = plt.gca()
    ax.set_title('SHAP Summary Plot (Beeswarm)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def plot_feature_importance_by_category(
    importance_df,
    category_summary,
    figsize=(14, 10),
    top_n_features=15
):
    """
    Create dual-panel visualization:
    - Panel 1: Category-level importance
    - Panel 2: Top features colored by category
    
    Args:
        importance_df: DataFrame with columns [feature, importance, category]
        category_summary: DataFrame with columns [category, importance, num_features, importance_pct]
        figsize: tuple of (width, height)
        top_n_features: number of top features in panel 2
    
    Returns:
        matplotlib.figure.Figure object
    """
    # Filter significant categories (>= 0.5% importance)
    significant_cats = category_summary[category_summary['importance_pct'] >= 0.5].copy()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors for categories
    colors = sns.color_palette('Set2', n_colors=len(significant_cats))
    colors_dict = dict(zip(significant_cats['category'], colors))
    
    # ========== Panel 1: Category Summary ==========
    ax1 = axes[0]
    y_pos = np.arange(len(significant_cats))
    bars = ax1.barh(y_pos, significant_cats['importance_pct'], color=colors)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, significant_cats['importance_pct'])):
        width = bar.get_width()
        ax1.text(
            width + 0.5, i, f'{pct:.1f}%',
            ha='left', va='center', fontsize=9
        )
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(significant_cats['category'], fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Feature Importance by Category',
        fontsize=13, fontweight='bold'
    )
    ax1.set_xlim(0, max(significant_cats['importance_pct']) * 1.15)
    ax1.grid(axis='x', alpha=0.3)
    
    # ========== Panel 2: Top Features by Category ==========
    ax2 = axes[1]
    
    # Get top features from significant categories
    top_features_data = []
    for cat in significant_cats['category']:
        cat_features = importance_df[importance_df['category'] == cat]
        for _, row in cat_features.nlargest(3, 'importance').iterrows():
            top_features_data.append({
                'category': cat,
                'feature': row['feature'],
                'importance': row['importance']
            })
    
    df_top = pd.DataFrame(top_features_data).nlargest(top_n_features, 'importance')
    
    # Plot
    feature_colors = [colors_dict.get(cat, 'gray') for cat in df_top['category']]
    y_pos = np.arange(len(df_top))
    
    ax2.barh(y_pos, df_top['importance'], color=feature_colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_top['feature'], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax2.set_title(
        f'Top {top_n_features} Most Important Features\n(Colored by Category)',
        fontsize=13, fontweight='bold'
    )
    ax2.grid(axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=colors_dict[cat], label=cat)
        for cat in significant_cats['category']
    ]
    ax2.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=9,
        framealpha=0.9
    )
    
    plt.tight_layout()
    return fig


def plot_shap_waterfall(
    shap_values,
    expected_value,
    feature_values,
    feature_names,
    sample_idx=0,
    max_display=10,
    figsize=(10, 6)
):
    """
    Create SHAP waterfall plot for a specific prediction
    
    Args:
        shap_values: numpy array of SHAP values
        expected_value: base value (model's expected output)
        feature_values: DataFrame or array of feature values
        feature_names: list of feature names
        sample_idx: index of sample to explain
        max_display: maximum number of features to display
        figsize: tuple of (width, height)
    
    Returns:
        matplotlib.figure.Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get SHAP values and features for specific sample
    sample_shap = shap_values[sample_idx]
    
    if isinstance(feature_values, pd.DataFrame):
        sample_features = feature_values.iloc[sample_idx].values
    else:
        sample_features = feature_values[sample_idx]
    
    # Create waterfall data
    shap_data = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sample_shap,
        'feature_value': sample_features
    })
    
    # Sort by absolute SHAP value
    shap_data['abs_shap'] = np.abs(shap_data['shap_value'])
    shap_data = shap_data.sort_values('abs_shap', ascending=False)
    
    # Take top features
    top_data = shap_data.head(max_display)
    
    # Prepare waterfall
    cumsum = expected_value
    positions = []
    values = []
    colors_list = []
    labels = []
    
    for _, row in top_data.iterrows():
        positions.append(cumsum)
        values.append(row['shap_value'])
        colors_list.append('green' if row['shap_value'] > 0 else 'red')
        
        # Create label with feature name and value
        labels.append(f"{row['feature']}\n= {row['feature_value']:.2f}")
        cumsum += row['shap_value']
    
    # Plot
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, left=positions, color=colors_list, alpha=0.7)
    
    # Add base value and final prediction lines
    ax.axvline(expected_value, color='black', linestyle='--', linewidth=2, label=f'Base value: {expected_value:.2f}')
    ax.axvline(cumsum, color='blue', linestyle='--', linewidth=2, label=f'Prediction: {cumsum:.2f}')
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Model Output (log units)', fontsize=11, fontweight='bold')
    ax.set_title(
        f'SHAP Waterfall Plot - Sample {sample_idx}',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_shap_dependence(
    shap_values,
    X_sample,
    feature_name,
    interaction_feature=None,
    figsize=(10, 6)
):
    """
    Create SHAP dependence plot showing relationship between
    feature value and its SHAP value
    
    Args:
        shap_values: numpy array of SHAP values
        X_sample: DataFrame of feature values
        feature_name: name of feature to plot
        interaction_feature: optional feature for interaction coloring
        figsize: tuple of (width, height)
    
    Returns:
        matplotlib.figure.Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create dependence plot
    shap.dependence_plot(
        feature_name,
        shap_values,
        X_sample,
        interaction_index=interaction_feature,
        show=False,
        ax=ax
    )
    
    # Customize
    ax.set_title(
        f'SHAP Dependence Plot: {feature_name}',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel(f'{feature_name} (Feature Value)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'SHAP Value for {feature_name}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_local_explanation_table(
    feature_names,
    feature_values,
    shap_values,
    top_n=5
):
    """
    Create tables showing top features increasing/decreasing prediction
    
    Args:
        feature_names: list of feature names
        feature_values: array of feature values for one sample
        shap_values: array of SHAP values for one sample
        top_n: number of top features to show
    
    Returns:
        tuple of (increasing_df, decreasing_df)
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_values,
        'SHAP': shap_values
    })
    
    # Top increasing factors
    increasing = df.nlargest(top_n, 'SHAP')[['Feature', 'Value', 'SHAP']]
    increasing.columns = ['Feature', 'Feature Value', 'SHAP Impact']
    
    # Top decreasing factors
    decreasing = df.nsmallest(top_n, 'SHAP')[['Feature', 'Value', 'SHAP']]
    decreasing.columns = ['Feature', 'Feature Value', 'SHAP Impact']
    
    return increasing, decreasing
