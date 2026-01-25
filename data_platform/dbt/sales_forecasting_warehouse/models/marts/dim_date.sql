{{ config(materialized='table') }}

with bounds as (
    select
        least(
            (select min(date) from {{ source('intermediate', 'int_active_sales') }}),
            (select min(date) from {{ source('intermediate', 'weather_features') }})
        ) as start_date,
        greatest(
            (select max(date) from {{ source('intermediate', 'int_active_sales') }}),
            (select max(date) from {{ source('intermediate', 'weather_features') }})
        ) as end_date
),

date_spine as (
    select 
        generate_series(
            (select start_date from bounds),
            (select end_date from bounds),
            '1 day'::interval
        )::date as date
),

attrs as (
    select *
    from {{ source('intermediate', 'int_date_features') }}
)

select
    d.date,
    extract(year from d.date)::int  as year,
    extract(month from d.date)::int as month,
    extract(day from d.date)::int   as day,
    extract(dow from d.date)::int   as day_of_week,

    -- Use attributes from int_date_features if they exist for this date
    coalesce(a.quarter, extract(quarter from d.date)::int) as quarter,
    coalesce(a.is_weekend, case when extract(dow from d.date) in (0,6) then 1 else 0 end) as is_weekend,
    coalesce(a.is_holiday, 0) as is_holiday,
    coalesce(a.is_blackfriday, 0) as is_blackfriday,
    coalesce(a.season_winter, 0) as season_winter,
    coalesce(a.season_spring, 0) as season_spring,
    coalesce(a.season_summer, 0) as season_summer,
    coalesce(a.season_fall,   0) as season_fall
from date_spine d
left join attrs a
  on d.date = a.date
