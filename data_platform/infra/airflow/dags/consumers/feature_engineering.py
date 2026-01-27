from airflow import DAG
from airflow.decorators import task
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
import sys
import os

# Ensure dags folder is in path to import datasets
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_CURATED_SALES, DS_CURATED_WEATHER, DS_FEATURE_STORE

def attach_features_metadata(context, result=None):
    """
    Attach metadata to the Feature Store dataset event.
    """
    context["outlet_events"][DS_FEATURE_STORE].extra = {
        "run_date": context["params"].get("run_date", context["ds"]),
        "source": "spark_feature_engineering"
    }

@task
def resolve_upstream_run_date(**context):
    """
    Resolve run_date from triggering dataset events (Airflow 2.10+).
    Key must be Dataset object, not URI string.
    """
    events = context.get("triggering_dataset_events", {})
    # Use Dataset object as key, not URI string
    sales_events = events.get(DS_CURATED_SALES, [])
    weather_events = events.get(DS_CURATED_WEATHER, [])
    
    # Get the latest event's extra data
    if sales_events:
        sales_date = (sales_events[-1].extra or {}).get("run_date")
        if sales_date:
            return sales_date
    
    return context["ds"]

with DAG(
    dag_id="consumer_feature_engineering",
    start_date=datetime(2024, 1, 1),
    # Triggered when BOTH are updated
    schedule=[DS_CURATED_SALES, DS_CURATED_WEATHER],
    catchup=False,
    tags=["layer:feature_store"],
) as dag:

    target_date = resolve_upstream_run_date()

    # Join Sales + Weather -> Create Features (Lags, Rolling, etc.)
    build_features = SparkSubmitOperator(
        task_id="spark_build_features",
        # Assuming this job exists or will be created
        application="/opt/spark/jobs/intermediate/join_features_pipeline.py",
        conn_id="spark_default",
        application_args=["--date", target_date],
        outlets=[DS_FEATURE_STORE],
        post_execute=attach_features_metadata,
        conf={
            "spark.submit.deployMode": "client",
        }
    )
