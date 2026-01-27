from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from pendulum import datetime
from cosmos import DbtDag, ProjectConfig, ProfileConfig, ExecutionConfig
import sys
import os

# Ensure dags folder is in path to import datasets
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_FEATURE_STORE

DBT_PROJECT_PATH = "/opt/airflow/dags/dbt/sales_forecasting"
DBT_EXECUTABLE_PATH = "/opt/airflow/dbt_venv/bin/dbt"

with DAG(
    dag_id="consumer_dw_serving",
    start_date=datetime(2024, 1, 1),
    schedule=[DS_FEATURE_STORE],
    catchup=False,
    tags=["layer:dw_serving"],
) as dag:

    # 1. Load Feature Store (Parquet) -> Postgres Staging
    load_to_dw = SparkSubmitOperator(
        task_id="spark_load_to_postgres",
        application="/opt/spark/jobs/load_to_postgres.py",
        conn_id="spark_default",
        application_args=[
            "--source", "s3a://datalake/feature_store/sales_forecast",
            "--target", "raw.stg_sales_forecast",
            "--mode", "overwrite"
        ],
        conf={
            "spark.submit.deployMode": "client",
        }
    )

    # 2. dbt build (via Cosmos)
    from cosmos import DbtTaskGroup, ProjectConfig, ProfileConfig, ExecutionConfig
    from cosmos.profiles import PostgresUserPasswordProfileMapping

    dbt_marts = DbtTaskGroup(
        group_id="dbt_marts_processing",
        project_config=ProjectConfig(DBT_PROJECT_PATH),
        profile_config=ProfileConfig(
            profile_name="sales_forecasting",
            target_name="dev",
            profile_mapping=PostgresUserPasswordProfileMapping(
                conn_id="postgres_dw", # Airflow Connection ID for the Data Warehouse
                profile_args={"schema": "marts"},
            ),
        ),
        execution_config=ExecutionConfig(dbt_executable_path=DBT_EXECUTABLE_PATH),
    )

    load_to_dw >> dbt_marts
