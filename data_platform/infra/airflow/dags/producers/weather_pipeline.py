from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_INTER_WEATHER, URI_INTER_WEATHER, DS_STG_DATA


def attach_run_date(context, result=None):
    logical_date = context.get("logical_date")
    run_date = context.get("ds") or (logical_date.to_date_string() if logical_date else None)
    batch_id = context.get("run_id")

    context["outlet_events"][DS_INTER_WEATHER].extra = {
        "run_date": run_date,
        "batch_id": batch_id
    }


with DAG(
    dag_id="producer_curate_weather",
    start_date=datetime(2024, 1, 1),
    schedule=[DS_STG_DATA],
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    params={"run_date": None},
    tags=["domain:weather", "layer:curate"],
) as dag:

    SparkSubmitOperator(
        task_id="spark_curate_weather",
        application="/opt/spark/jobs/intermediate/weather_features_pipeline.py",
        conn_id="spark_default",
        properties_file="/opt/spark/conf/spark-defaults.conf",
        application_args=["--date", "{{ params.run_date or ds }}"],
        outlets=[DS_INTER_WEATHER],
        post_execute=attach_run_date,
        jars="/opt/airflow/jars/hadoop-aws-3.3.4.jar,/opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar,/opt/airflow/jars/postgresql-42.7.0.jar",
        conf={
            "spark.submit.deployMode": "client",
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
        },
    )
