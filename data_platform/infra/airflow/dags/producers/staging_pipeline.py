"""
Producer: Ingest Raw Data
Layer   : Ingestion
Trigger : @daily (schedule-based)
Output  : DS_RAW_PARQUET_READY
Mô tả   : Đọc CSV từ MinIO raw zone, convert sang Parquet.
          Đây là bước đầu tiên trong toàn bộ lakehouse pipeline.
          Staging transform (stg_*) do dbt sales_forecasting_lakehouse đảm nhiệm.
"""
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_RAW_PARQUET_READY


def emit_raw_parquet_ready(context, result=None):
    logical_date = context.get("logical_date")
    context["outlet_events"][DS_RAW_PARQUET_READY].extra = {
        "run_date": context.get("ds") or (logical_date.to_date_string() if logical_date else None),
        "batch_id": context.get("run_id"),
    }


with DAG(
    dag_id="producer_ingest_raw",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["layer:ingestion", "domain:all"],
) as dag:

    SparkSubmitOperator(
        task_id="ingest_raw_csv_to_parquet",
        application="/opt/spark/jobs/staging/ingest_raw_to_parquet.py",
        conn_id="spark_default",
        properties_file="/opt/spark/conf/spark-defaults.conf",
        outlets=[DS_RAW_PARQUET_READY],
        post_execute=emit_raw_parquet_ready,
        jars="/opt/airflow/jars/hadoop-aws-3.3.4.jar,/opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar,/opt/airflow/jars/iceberg-spark-runtime-3.5_2.12-1.10.1.jar,/opt/airflow/jars/nessie-spark-extensions-3.5_2.12-0.107.2.jar",
        conf={
            "spark.submit.deployMode": "client",
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
            "spark.driver.extraClassPath": "/opt/airflow/jars/*",
        },
    )
