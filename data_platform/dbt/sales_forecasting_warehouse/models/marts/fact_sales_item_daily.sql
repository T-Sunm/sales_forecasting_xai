{{ config(materialized='table') }}

select
  date,
  store_id,
  item_id,
  units,
  log_units
from {{ source('intermediate', 'int_active_sales') }}
