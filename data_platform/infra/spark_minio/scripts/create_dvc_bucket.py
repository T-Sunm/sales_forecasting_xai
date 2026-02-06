import os
from minio import Minio
from minio.error import S3Error

ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET_NAME = os.getenv("DVC_BUCKET", "dvc-store")

def main():
    client = Minio(ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=SECURE)

    if client.bucket_exists(BUCKET_NAME):
        print(f"Bucket '{BUCKET_NAME}' already exists.")
    else:
        client.make_bucket(BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created successfully.")

if __name__ == "__main__":
    try:
        main()
    except S3Error as e:
        raise SystemExit(e)
