from pathlib import Path
import os
import mlflow
from dotenv import load_dotenv

def setup_mlflow():
    """Setup MLflow tracking URI and other configurations from .env using absolute paths"""
    # Find .env in project root (ml/utils/mlflow_utils.py -> sales_forecasting_xai/.env)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    
    env_loaded = False
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    # Force HTTP server if the env is set to a local store (sqlite/file) or empty
    if not tracking_uri or tracking_uri.startswith("sqlite:") or tracking_uri.startswith("file:"):
        tracking_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Ensure S3 endpoint for MinIO is set for artifacts (required for tracking server/client to know where to upload)
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if s3_endpoint:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    
    print(f"MLflow Setup Info:")
    print(f"  - Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"  - Environment Loaded From: {env_path if env_loaded else 'None (Defaults)'}")
    
    return tracking_uri
