"""
Producer: EWMA Sales Features
Layer   : Intermediate (Spark-only)
Trigger : DS_RAW_PARQUET_READY
Output  : DS_EWMA_FEATURES_READY
Mô tả   : Chạy sales_features_pipeline.py để tạo int_sales_with_ewma (applyInPandas
          EWMA — không thể thay bằng Spark SQL window function).
          Các intermediate khác (lags, rolling, aggregates, date) do dbt đảm nhiệm.
"""
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_RAW_PARQUET_READY, DS_EWMA_FEATURES_READY


def emit_ewma_ready(context, result=None):
    logical_date = context.get("logical_date")
    context["outlet_events"][DS_EWMA_FEATURES_READY].extra = {
        "run_date": context.get("ds") or (logical_date.to_date_string() if logical_date else None),
        "batch_id": context.get("run_id"),
    }


with DAG(
    dag_id="producer_ewma_features",
    start_date=datetime(2024, 1, 1),
    schedule=[DS_RAW_PARQUET_READY],
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["layer:intermediate", "domain:sales", "tool:spark"],
) as dag:

    SparkSubmitOperator(
        task_id="compute_ewma_sales_features",
        application="/opt/spark/jobs/intermediate/sales_features_pipeline.py",
        conn_id="spark_default",
        properties_file="/opt/spark/conf/spark-defaults.conf",
        outlets=[DS_EWMA_FEATURES_READY],
        post_execute=emit_ewma_ready,
        jars="/opt/airflow/jars/hadoop-aws-3.3.4.jar,/opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar,/opt/airflow/jars/iceberg-spark-runtime-3.5_2.12-1.10.1.jar,/opt/airflow/jars/nessie-spark-extensions-3.5_2.12-0.107.2.jar",
        conf={
            "spark.submit.deployMode": "client",
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
            "spark.driver.extraClassPath": "/opt/airflow/jars/*",
        },
    )
