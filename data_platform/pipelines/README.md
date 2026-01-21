# Pipelines & Orchestration

Thư mục này chứa các workflow orchestration definitions.

## Airflow

Sử dụng Airflow để orchestrate các Spark jobs và dbt runs.

### DAG Flow

```
1. Load Raw Data → MinIO (Bronze)
2. Spark Staging Jobs → MinIO (Silver)
3. Spark Intermediate Jobs → MinIO (Gold)
4. Load Gold → PostgreSQL
5. dbt Run → Star Schema (Marts)
```

### Example DAG Structure

```python
# sales_forecasting_dag.py
with DAG('sales_forecasting') as dag:
    load_raw = PythonOperator(...)
    spark_staging = SparkSubmitOperator(...)
    spark_intermediate = SparkSubmitOperator(...)
    load_to_postgres = PythonOperator(...)
    dbt_run = BashOperator(...)
    
    load_raw >> spark_staging >> spark_intermediate >> load_to_postgres >> dbt_run
```
