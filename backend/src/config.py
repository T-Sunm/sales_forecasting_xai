from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SHARED_DIR = BASE_DIR / "shared"

DATA_DIR = SHARED_DIR / "data"
MODELS_DIR = SHARED_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = SHARED_DIR / "logs"

DATA_RAW_DIR = DATA_DIR / "data_raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.parquet"
VALID_DATA_PATH = DATA_PROCESSED_DIR / "valid.parquet"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.parquet"

FEATURE_DATA_PATH = TRAIN_DATA_PATH

LGBM_MODELS_PKL = MODELS_DIR / "lgbm_baseline.pkl"
FEATURE_STATS_JSON = MODELS_DIR / "feature_stats.json"

WEATHER_DATA_CSV = DATA_DIR / "weather_data.csv"
SALES_2016_CSV = DATA_DIR / "2016_sales.csv"
SALES_2017_CSV = DATA_DIR / "2017_sales.csv"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

DB_SCHEMA = os.getenv("DB_SCHEMA", "marts")

FEATURE_COUNT = 88
DATE_FORMAT = "%Y-%m-%d"

COL_STORE_ID = "store_id"
COL_ITEM_ID = "item_id"
COL_UNITS = "units"
COL_DATE = "date"

API_HOST = "0.0.0.0"
API_PORT = 8000
API_NAME = "Sales Forecasting API"
API_VERSION = "1.0.0"

# Database Configuration (PostgreSQL)
PG_DSN = os.getenv("DATABASE_URL", "postgresql://postgres:changeme@localhost:5432/sales_forecasting")

SECRET_KEY = os.getenv("SECRET_KEY", "my-super-secret-key-2025")
CORS_ORIGINS = ["*"]
RATE_LIMIT_PER_MINUTE = 60
MAX_RECURSIVE_FORECAST_DAYS = 365