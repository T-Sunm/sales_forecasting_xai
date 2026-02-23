from __future__ import annotations

import subprocess
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

_NAMESPACES = ["nessie.default", "nessie.analytics"]

_PYHIVE_CODE = """
from pyhive import hive

conn = hive.Connection(host="spark-thrift", port=10001, username="airflow")
cur = conn.cursor()
for ns in {namespaces!r}:
    cur.execute(f"CREATE NAMESPACE IF NOT EXISTS {{ns}}")
cur.execute("SHOW NAMESPACES IN nessie")
print("Namespaces:", cur.fetchall())
cur.close()
conn.close()
""".format(namespaces=_NAMESPACES)


def _bootstrap():
    subprocess.run(
        ["/opt/airflow/dbt_venv/bin/python", "-c", _PYHIVE_CODE],
        check=True,
        capture_output=False,
    )


with DAG(
    dag_id="bootstrap_nessie_namespaces",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["infra"],
) as dag:
    PythonOperator(
        task_id="create_nessie_namespaces",
        python_callable=_bootstrap,
        retries=2,
    )
