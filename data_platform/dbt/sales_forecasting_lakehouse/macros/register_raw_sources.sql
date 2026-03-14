{% macro register_raw_sources() %}
  {% set statements = [
    "CREATE SCHEMA IF NOT EXISTS spark_catalog.raw_parquet",
    "CREATE TABLE IF NOT EXISTS spark_catalog.raw_parquet.train         USING parquet LOCATION 's3://datalake/staging/parquet/train'",
    "CREATE TABLE IF NOT EXISTS spark_catalog.raw_parquet.weather       USING parquet LOCATION 's3://datalake/staging/parquet/weather'",
    "CREATE TABLE IF NOT EXISTS spark_catalog.raw_parquet.key           USING parquet LOCATION 's3://datalake/staging/parquet/key'",
    "CREATE TABLE IF NOT EXISTS spark_catalog.raw_parquet.holidays      USING parquet LOCATION 's3://datalake/staging/parquet/holidays'",
    "CREATE TABLE IF NOT EXISTS spark_catalog.raw_parquet.blackfriday   USING parquet LOCATION 's3://datalake/staging/parquet/blackfriday'",
    "CREATE TABLE IF NOT EXISTS spark_catalog.raw_parquet.int_sales_with_ewma USING parquet LOCATION 's3://datalake/intermediate/int_sales_with_ewma'",
  ] %}
  {% for stmt in statements %}
    {% do run_query(stmt) %}
  {% endfor %}
{% endmacro %}
