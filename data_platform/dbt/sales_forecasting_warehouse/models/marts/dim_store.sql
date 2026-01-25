{{ config(materialized='table') }}

select distinct
  store_id,
  station_id
from {{ source('raw', 'stg_key') }}
