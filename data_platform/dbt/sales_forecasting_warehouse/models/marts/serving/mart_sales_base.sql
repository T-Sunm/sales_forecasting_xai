-- models/marts/serving/mart_sales_base.sql
{{ config(materialized='table') }}

select 
    date, 
    store_id, 
    item_id, 
    units
from {{ ref('fact_sales_item_daily') }}
