from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from cosmos import DbtTaskGroup, ProjectConfig, ProfileConfig, ExecutionConfig, RenderConfig
from cosmos.constants import TestBehavior
from cosmos.profiles import PostgresUserPasswordProfileMapping
from pendulum import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import DS_MART_FEATURES

DBT_PROJECT_PATH = "/opt/airflow/dags/dbt/sales_forecasting_warehouse"
DBT_EXECUTABLE_PATH = "/opt/airflow/dbt_venv/bin/dbt"


with DAG(
    dag_id="consumer_dw_serving",
    start_date=datetime(2024, 1, 1),
    schedule=[DS_MART_FEATURES],
    catchup=False,
    tags=["layer:serving"],
) as dag:

    load_to_dw = SparkSubmitOperator(
        task_id="spark_load_to_postgres",
        application="/opt/spark/jobs/load_to_postgres.py",
        conn_id="spark_default",
        properties_file="/opt/spark/conf/spark-defaults.conf",
        jars="/opt/airflow/jars/hadoop-aws-3.3.4.jar,/opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar,/opt/airflow/jars/postgresql-42.7.0.jar",
        conf={
            "spark.submit.deployMode": "client",
            "spark.driver.host": "airflow-worker",
            "spark.driver.bindAddress": "0.0.0.0",
        },
        env_vars={
            "PG_HOST": "postgres_container",
            "PG_USER": "postgres",
            "PG_PASS": "changeme",
            "PG_DB": "sales_forecasting",
        },
    )

    dbt_marts = DbtTaskGroup(
        group_id="dbt_marts",
        project_config=ProjectConfig(DBT_PROJECT_PATH),
        profile_config=ProfileConfig(
            profile_name="sales_forecasting",
            target_name="dev",
            profile_mapping=PostgresUserPasswordProfileMapping(
                conn_id="postgres_dw",
                profile_args={"schema": "marts"},
            ),
        ),
        execution_config=ExecutionConfig(dbt_executable_path=DBT_EXECUTABLE_PATH),
        render_config=RenderConfig(test_behavior=TestBehavior.AFTER_ALL),
    )

    load_to_dw >> dbt_marts
