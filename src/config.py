from pathlib import Path
import os

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent  # Root của project
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
DATA_RAW_DIR = DATA_DIR / "data_raw"
DATA_PROCESSED_DIR = DATA_DIR / "data_processed"

# Specific files
WEATHER_KEY_STORE_CSV = DATA_PROCESSED_DIR / "weather_key_store_merged.csv"
FEATURE_ENGINEERED_FEATHER = DATA_PROCESSED_DIR / "feature_engineered_data_88_features.feather"

# Model files
LGBM_MODELS_PKL = MODELS_DIR / "lgbm_models_dict.pkl"
FEATURE_STATS_JSON = MODELS_DIR / "feature_stats.json"

# Generated data files (from data_generator)
WEATHER_DATA_CSV = DATA_DIR / "weather_data.csv"
SALES_2016_CSV = DATA_DIR / "2016_sales.csv"
SALES_2017_CSV = DATA_DIR / "2017_sales.csv"

# API Keys (từ .env file)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# App settings
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

# Feature names
FEATURE_COUNT = 88  

# Date formats
DATE_FORMAT = "%Y-%m-%d"

# Column names
COL_STORE_NBR = "store_nbr"
COL_ITEM_NBR = "item_nbr"
COL_UNITS = "units"
COL_DATE = "date"