-- models/marts/serving/mart_store_day.sql
{{ config(materialized='table') }}

select
    date,
    store_id,
    sum(units) as total_units,
    count(*) as sales_records,
    sum(units)::numeric / nullif(count(*), 0) as avg_units_per_record
from {{ ref('fact_sales_item_daily') }}
group by 1, 2
