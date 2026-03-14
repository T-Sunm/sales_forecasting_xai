"""
Consumer: Lakehouse Pipeline (dbt)
Layer   : Staging → Intermediate → Mart
Trigger : DS_RAW_PARQUET_READY + DS_EWMA_FEATURES_READY
Output  : DS_LAKEHOUSE_MART_READY
Mô tả   : Chạy toàn bộ dbt project sales_forecasting_lakehouse qua Cosmos DbtTaskGroup.
          Kết quả là Iceberg tables trong Nessie catalog:
            - nessie.default.*  : staging + intermediate
            - nessie.analytics.* : mart_sales_forecast
"""
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from cosmos import DbtTaskGroup, ProjectConfig, ProfileConfig, ExecutionConfig, RenderConfig
from cosmos.constants import TestBehavior
from pendulum import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_RAW_PARQUET_READY, DS_EWMA_FEATURES_READY, DS_LAKEHOUSE_MART_READY

DBT_PROJECT_PATH    = "/opt/airflow/dags/dbt/sales_forecasting_lakehouse"
DBT_EXECUTABLE_PATH = "/opt/airflow/dbt_venv/bin/dbt"


with DAG(
    dag_id="consumer_lakehouse_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=[DS_RAW_PARQUET_READY, DS_EWMA_FEATURES_READY],
    catchup=False,
    tags=["layer:lakehouse", "domain:all", "tool:dbt"],
) as dag:

    dbt_lakehouse = DbtTaskGroup(
        group_id="dbt_sales_forecasting_lakehouse",
        project_config=ProjectConfig(DBT_PROJECT_PATH),
        profile_config=ProfileConfig(
            profile_name="sales_forecasting_lakehouse",
            target_name="dev",
            profiles_yml_filepath=f"{DBT_PROJECT_PATH}/profiles.yml",
        ),
        execution_config=ExecutionConfig(dbt_executable_path=DBT_EXECUTABLE_PATH),
        render_config=RenderConfig(test_behavior=TestBehavior.AFTER_ALL),
    )

    lakehouse_mart_ready = EmptyOperator(
        task_id="lakehouse_mart_ready",
        outlets=[DS_LAKEHOUSE_MART_READY],
    )

    dbt_lakehouse >> lakehouse_mart_ready
