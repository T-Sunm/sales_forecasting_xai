from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
import sys
import os

# Ensure dags folder is in path to import datasets
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_CURATED_SALES

def attach_run_date_to_dataset(context, result=None):
    """
    Airflow 2.10+ hook to attach metadata to the dataset event.
    """
    context["outlet_events"][DS_CURATED_SALES].extra = {
        "run_date": context["ds"],
        "batch_id": context["run_id"]
    }

with DAG(
    dag_id="producer_curate_sales",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["domain:sales", "layer:curate"],
) as dag:

    # Task for Spark curation
    curate_sales = SparkSubmitOperator(
        task_id="spark_curate_sales",
        application="/opt/spark/jobs/staging/sales_features_pipeline.py",
        conn_id="spark_default",
        application_args=["--date", "{{ ds }}"],
        outlets=[DS_CURATED_SALES],
        post_execute=attach_run_date_to_dataset,
        conf={
            "spark.submit.deployMode": "client",
        }
    )
