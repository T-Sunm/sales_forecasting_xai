{{ config(materialized='table', file_format='iceberg') }}

select distinct
    item_id
from {{ ref('stg_sales') }}
