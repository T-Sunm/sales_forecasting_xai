from airflow import DAG
from airflow.decorators import task
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import (
    DS_INTER_SALES, DS_INTER_WEATHER, DS_MART_FEATURES,
    URI_INTER_SALES, URI_MART_FEATURES
)


def attach_metadata(context, result=None):
    logical_date = context.get("logical_date")
    run_date = context.get("ds") or (logical_date.to_date_string() if logical_date else None)
    
    context["outlet_events"][DS_MART_FEATURES].extra = {
        "run_date": context["params"].get("run_date", run_date),
        "source": "spark_join_features"
    }


@task
def resolve_run_date(triggering_asset_events=None, **context):
    events = triggering_asset_events or {}
    
    for asset, asset_events in events.items():
        # Kiểm tra URI của asset để tìm đúng nguồn sales
        if getattr(asset, "uri", None) == URI_INTER_SALES and asset_events:
            run_date = (asset_events[-1].extra or {}).get("run_date")
            if run_date:
                return run_date

    return context["ds"]


with DAG(
    dag_id="consumer_feature_engineering",
    start_date=datetime(2024, 1, 1),
    schedule=[DS_INTER_SALES, DS_INTER_WEATHER],
    catchup=False,
    tags=["layer:mart"],
) as dag:

    target_date = resolve_run_date()

    SparkSubmitOperator(
        task_id="spark_join_features",
        application="/opt/spark/jobs/intermediate/join_features_pipeline.py",
        conn_id="spark_default",
        properties_file="/opt/spark/conf/spark-defaults.conf",
        application_args=["--date", target_date],
        outlets=[DS_MART_FEATURES],
        post_execute=attach_metadata,
        jars="/opt/airflow/jars/hadoop-aws-3.3.4.jar,/opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar,/opt/airflow/jars/postgresql-42.7.0.jar",
        conf={
            "spark.submit.deployMode": "client",
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
        },
    )
