{{ config(materialized='table') }}

select distinct
  item_id
from {{ source('intermediate', 'int_active_sales') }}
