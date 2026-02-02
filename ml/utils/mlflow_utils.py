import os
import mlflow
from dotenv import load_dotenv

def setup_mlflow():
    """Setup MLflow tracking URI and other configurations from .env"""
    # Try to load from root .env or ml/.env
    if os.path.exists("../.env"):
        load_dotenv("../.env")
    elif os.path.exists(".env"):
        load_dotenv(".env")
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Ensure S3 endpoint for MinIO is set for artifacts
    if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    
    return tracking_uri
