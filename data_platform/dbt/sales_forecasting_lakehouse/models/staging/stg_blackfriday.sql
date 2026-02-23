{{ config(schema='default') }}

select to_date(date) as date
from parquet.`s3a://datalake/staging/parquet/blackfriday`
