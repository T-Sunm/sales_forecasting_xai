import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SHARED_DIR = BASE_DIR / "shared"

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

TEMP_DIR = BASE_DIR / "frontend" / "temp"
FIGURES_DIR = SHARED_DIR.parent / "figures"

DATA_DIR = SHARED_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.parquet"
VALID_DATA_PATH = DATA_PROCESSED_DIR / "valid.parquet"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.parquet"

# Legacy name for compatibility during transition
FEATURE_DATA_PATH = TRAIN_DATA_PATH
FEATURE_STATS_JSON = SHARED_DIR / "models" / "feature_stats.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

COL_STORE_ID = "store_id"
COL_ITEM_ID = "item_id"
COL_UNITS = "units"
COL_DATE = "date"
