{{ config(materialized='table', file_format='iceberg') }}

select distinct
    store_id,
    station_id
from {{ ref('stg_key') }}
