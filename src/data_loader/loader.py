import json
import pickle

import pandas as pd
import streamlit as st

from src.config import (
    LGBM_MODELS_PKL,
    FEATURE_STATS_JSON,
    WEATHER_KEY_STORE_CSV,
    FEATURE_ENGINEERED_FEATHER,
)


@st.cache_resource
def load_model():
    """Load the trained sales forecast models dictionary"""
    try:
        with open(LGBM_MODELS_PKL, "rb") as file:
            models_dict = pickle.load(file)
        return models_dict
    except FileNotFoundError:
        st.error(
            f"Model file not found. Please ensure '{LGBM_MODELS_PKL}' exists."
        )
        return None


@st.cache_resource
def load_feature_stats():
    """Load feature statistics used for normalization"""
    try:
        with open(FEATURE_STATS_JSON, "r") as file:
            feature_stats = json.load(file)
        return feature_stats
    except FileNotFoundError:
        st.error(
            f"Feature stats file not found. Please ensure '{FEATURE_STATS_JSON}' exists."
        )
        return {}


@st.cache_data
def load_data():
    """Load preprocessed sales data"""
    try:
        # Load the preprocessed data
        df = pd.read_csv(WEATHER_KEY_STORE_CSV)

        # Filter out kaggle test data
        if "is_kaggle_test" in df.columns:
            df = df[df["is_kaggle_test"] == False]

        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df
    except FileNotFoundError:
        st.error(
            f"Data file not found. Please ensure '{WEATHER_KEY_STORE_CSV}' exists."
        )
        # Return empty DataFrame with expected columns as fallback
        return pd.DataFrame(columns=["date", "store", "sales"])


@st.cache_data
def load_feature_engineered_data():
    """Load feature engineered data with extended features for predictions"""
    try:
        import pyarrow.feather as feather

        feature_engineered_data = feather.read_feather(
            FEATURE_ENGINEERED_FEATHER
        )
        return feature_engineered_data
    except Exception as e:
        st.error(f"Error loading feature engineered data: {str(e)}")
        st.info(
            f"Please ensure the file '{FEATURE_ENGINEERED_FEATHER}' exists."
        )
        return pd.DataFrame()


def preprocess_data(df, feature_stats=None):
    """Preprocess data for prediction (simplified version)"""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Extract date features if date column exists
    if "date" in processed_df.columns:
        processed_df["day_of_week"] = processed_df["date"].dt.dayofweek
        processed_df["day_of_month"] = processed_df["date"].dt.day
        processed_df["month"] = processed_df["date"].dt.month
        processed_df["year"] = processed_df["date"].dt.year
        processed_df["is_weekend"] = processed_df["day_of_week"].apply(
            lambda x: 1 if x >= 5 else 0
        )

    # Normalize numerical features if stats are provided
    if feature_stats:
        for feature, stats in feature_stats.items():
            if feature in processed_df.columns and "mean" in stats and "std" in stats:
                processed_df[feature] = (processed_df[feature] - stats["mean"]) / stats[
                    "std"
                ]

    return processed_df


def get_top_data_pair(df):
    """Find the (store_nbr, item_nbr) pair with the most records"""
    if df.empty or "store_nbr" not in df.columns or "item_nbr" not in df.columns:
        return None, None
        
    counts = (
        df.groupby(["store_nbr", "item_nbr"])
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )
    
    if counts.empty:
        return None, None
        
    top_row = counts.iloc[0]
    return int(top_row["store_nbr"]), int(top_row["item_nbr"])
