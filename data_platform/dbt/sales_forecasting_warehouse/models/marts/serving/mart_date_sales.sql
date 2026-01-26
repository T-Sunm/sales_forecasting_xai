-- models/marts/serving/mart_date_sales.sql
{{ config(materialized='table') }}

with daily_sales as (
    select 
        date, 
        sum(units) as total_units, 
        count(*) as sales_records
    from {{ ref('fact_sales_item_daily') }}
    group by 1
)
select
    d.date,
    d.day_of_week,
    d.month,
    d.is_holiday,
    d.is_blackfriday,
    coalesce(s.total_units, 0) as total_units,
    coalesce(s.sales_records, 0) as sales_records,
    coalesce(s.total_units, 0)::numeric / nullif(coalesce(s.sales_records, 0), 0) as avg_units_per_record
from {{ ref('dim_date') }} d
left join daily_sales s on d.date = s.date
