{{ config(materialized='table') }}

select distinct
    item_id
from {{ ref('stg_sales') }}
