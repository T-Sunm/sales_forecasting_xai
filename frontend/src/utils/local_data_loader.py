"""
Local Data Loader
Load data files for visualization purposes ONLY
Prediction logic should use Backend API
"""

import json
import pandas as pd
import streamlit as st
import pyarrow.feather as feather

from config import (
    FEATURE_STATS_JSON,
    WEATHER_KEY_STORE_CSV,
    FEATURE_ENGINEERED_FEATHER,
)


@st.cache_data
def load_data():
    """
    Load preprocessed sales data for visualization
    Used by: Historical Dashboard
    """
    try:
        df = pd.read_csv(WEATHER_KEY_STORE_CSV)

        # Filter out kaggle test data
        if "is_kaggle_test" in df.columns:
            df = df[df["is_kaggle_test"] == False]

        # Convert date column to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {WEATHER_KEY_STORE_CSV}")
        return pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "units"])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def load_feature_engineered_data():
    """
    Load feature engineered data
    Used by: XAI Explanations (SHAP analysis)
    """
    try:
        feature_engineered_data = feather.read_feather(FEATURE_ENGINEERED_FEATHER)
        return feature_engineered_data
    except Exception as e:
        st.error(f"Error loading feature engineered data: {str(e)}")
        st.info(f"File: {FEATURE_ENGINEERED_FEATHER}")
        return pd.DataFrame()


@st.cache_data
def load_feature_stats():
    """
    Load feature statistics (lightweight, keep local)
    Used for: Feature normalization info display
    """
    try:
        with open(FEATURE_STATS_JSON, "r") as file:
            feature_stats = json.load(file)
        return feature_stats
    except FileNotFoundError:
        st.warning(f"Feature stats not found: {FEATURE_STATS_JSON}")
        return {}
    except Exception as e:
        st.error(f"Error loading feature stats: {str(e)}")
        return {}
