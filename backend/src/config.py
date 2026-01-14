from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SHARED_DIR = BASE_DIR / "shared"

DATA_DIR = SHARED_DIR / "data"
MODELS_DIR = SHARED_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = SHARED_DIR / "logs"

DATA_RAW_DIR = DATA_DIR / "data_raw"
DATA_PROCESSED_DIR = DATA_DIR / "data_processed"

WEATHER_KEY_STORE_CSV = DATA_PROCESSED_DIR / "weather_key_store_merged.csv"
FEATURE_ENGINEERED_FEATHER = DATA_PROCESSED_DIR / "feature_engineered_data_88_features.feather"

LGBM_MODELS_PKL = MODELS_DIR / "lgbm_models_dict.pkl"
FEATURE_STATS_JSON = MODELS_DIR / "feature_stats.json"

WEATHER_DATA_CSV = DATA_DIR / "weather_data.csv"
SALES_2016_CSV = DATA_DIR / "2016_sales.csv"
SALES_2017_CSV = DATA_DIR / "2017_sales.csv"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

FEATURE_COUNT = 88
DATE_FORMAT = "%Y-%m-%d"

COL_STORE_NBR = "store_nbr"
COL_ITEM_NBR = "item_nbr"
COL_UNITS = "units"
COL_DATE = "date"

API_HOST = "0.0.0.0"
API_PORT = 8000
API_NAME = "Sales Forecasting API"
API_VERSION = "1.0.0"

SECRET_KEY = os.getenv("SECRET_KEY", "my-super-secret-key-2025")
CORS_ORIGINS = ["*"]
RATE_LIMIT_PER_MINUTE = 60
MAX_RECURSIVE_FORECAST_DAYS = 365