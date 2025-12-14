from pathlib import Path
import os

# ==================== 1. ĐƯỜNG DẪN FOLDER (PATH CONFIG) ====================

# Lấy đường dẫn gốc của dự án (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Các folder chính
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# Folder con trong data
DATA_RAW_DIR = DATA_DIR / "data_raw"
DATA_PROCESSED_DIR = DATA_DIR / "data_processed"

# ==================== 2. ĐƯỜNG DẪN FILE (FILE PATHS) ====================

# File dữ liệu
WEATHER_KEY_STORE_CSV = DATA_PROCESSED_DIR / "weather_key_store_merged.csv"
FEATURE_ENGINEERED_FEATHER = DATA_PROCESSED_DIR / "feature_engineered_data_88_features.feather"

# File model
LGBM_MODELS_PKL = MODELS_DIR / "lgbm_models_dict.pkl"
FEATURE_STATS_JSON = MODELS_DIR / "feature_stats.json"

# File output generated
WEATHER_DATA_CSV = DATA_DIR / "weather_data.csv"
SALES_2016_CSV = DATA_DIR / "2016_sales.csv"
SALES_2017_CSV = DATA_DIR / "2017_sales.csv"

# ==================== 3. CẤU HÌNH APP (APP SETTINGS) ====================

# API Keys (Lấy từ môi trường hoặc để trống)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Chế độ Debug
DEBUG_MODE = os.getenv("DEBUG", "True").lower() == "true"

# Thông số dữ liệu
FEATURE_COUNT = 88
DATE_FORMAT = "%Y-%m-%d"

# Tên cột (Column Names)
COL_STORE_NBR = "store_nbr"
COL_ITEM_NBR = "item_nbr"
COL_UNITS = "units"
COL_DATE = "date"

# ==================== 4. CẤU HÌNH API (SIMPLE API SETTINGS) ====================
# Phần này dùng cho FastAPI, chạy đơn giản không cần Class phức tạp

# Server info
API_HOST = "0.0.0.0"
API_PORT = 8000
API_NAME = "Sales Forecasting API"
API_VERSION = "1.0.0"

# Security (Đơn giản hóa: Dùng key mặc định nếu không set)
SECRET_KEY = os.getenv("SECRET_KEY", "my-super-secret-key-2025")

# CORS (Cho phép tất cả để dễ dev/test)
CORS_ORIGINS = ["*"]  

# Giới hạn Request (Rate Limit)
RATE_LIMIT_PER_MINUTE = 60

# Model Settings
MAX_RECURSIVE_FORECAST_DAYS = 365 # Giới hạn số ngày dự đoán đệ quy