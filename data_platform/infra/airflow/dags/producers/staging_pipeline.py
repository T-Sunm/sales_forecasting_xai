from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_STG_DATA


def attach_metadata(context, result=None):
    logical_date = context.get("logical_date")
    run_date = context.get("ds") or (logical_date.to_date_string() if logical_date else None)
    batch_id = context.get("run_id")
    
    context["outlet_events"][DS_STG_DATA].extra = {
        "run_date": run_date,
        "batch_id": batch_id
    }


with DAG(
    dag_id="producer_staging_layer",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["layer:staging", "domain:all"],
) as dag:

    ingest_raw = SparkSubmitOperator(
        task_id="spark_ingest_raw_to_parquet",
        application="/opt/spark/jobs/staging/ingest_raw_to_parquet.py",
        conn_id="spark_default",
        properties_file="/opt/spark/conf/spark-defaults.conf",
        jars="/opt/airflow/jars/hadoop-aws-3.3.4.jar,/opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar",
        conf={
            "spark.submit.deployMode": "client",
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
        },
    )

    transform_stg = SparkSubmitOperator(
        task_id="spark_staging_transform",
        application="/opt/spark/jobs/staging/staging_transform.py",
        conn_id="spark_default",
        properties_file="/opt/spark/conf/spark-defaults.conf",
        outlets=[DS_STG_DATA],
        post_execute=attach_metadata,
        jars="/opt/airflow/jars/hadoop-aws-3.3.4.jar,/opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar",
        conf={
            "spark.submit.deployMode": "client",
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
        },
    )

    ingest_raw >> transform_stg
