import os
import boto3
import mlflow
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv("../.env")

# Configs
ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
BUCKET = "mlflow"

def ensure_bucket():
    s3 = boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=KEY, aws_secret_access_key=SECRET)
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)
        print(f"Created bucket: {BUCKET}")

def run_test():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("connection-test")
    
    with mlflow.start_run(run_name="test-run"):
        mlflow.log_param("test_key", "test_value")
        
        with open("test.txt", "w") as f:
            f.write("test content")
        mlflow.log_artifact("test.txt")
        
        print(f"Test run successful. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    ensure_bucket()
    try:
        run_test()
    finally:
        if os.path.exists("test.txt"):
            os.remove("test.txt")
