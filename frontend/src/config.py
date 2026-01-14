import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SHARED_DIR = BASE_DIR / "shared"

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

TEMP_DIR = BASE_DIR / "frontend" / "temp"
FIGURES_DIR = SHARED_DIR.parent / "figures"

DATA_DIR = SHARED_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "data_processed"
WEATHER_KEY_STORE_CSV = DATA_PROCESSED_DIR / "weather_key_store_merged.csv"
FEATURE_ENGINEERED_FEATHER = DATA_PROCESSED_DIR / "feature_engineered_data_88_features.feather"

MODELS_DIR = SHARED_DIR / "models"
LGBM_MODELS_PKL = MODELS_DIR / "lgbm_models_dict.pkl"
FEATURE_STATS_JSON = MODELS_DIR / "feature_stats.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

COL_STORE_NBR = "store_nbr"
COL_ITEM_NBR = "item_nbr"
COL_UNITS = "units"
COL_DATE = "date"
