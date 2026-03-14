{{ config(materialized='table') }}

with bounds as (
    select
        least(
            (select min(date) from {{ ref('stg_sales') }}),
            (select min(date) from {{ ref('int_weather_features') }})
        ) as start_date,
        greatest(
            (select max(date) from {{ ref('stg_sales') }}),
            (select max(date) from {{ ref('int_weather_features') }})
        ) as end_date
),

date_spine as (
    select generate_series(
        (select start_date from bounds),
        (select end_date from bounds),
        '1 day'::interval
    )::date as date
),

attrs as (
    select distinct
        date,
        quarter,
        is_weekend,
        is_holiday,
        is_blackfriday,
        season_winter,
        season_spring,
        season_summer,
        season_fall
    from {{ ref('int_date_features') }}
)

select
    d.date,
    extract(year from d.date) as year,
    extract(month from d.date) as month,
    extract(day from d.date) as day,
    extract(dow from d.date) + 1 as day_of_week,

    coalesce(a.quarter, extract(quarter from d.date)) as quarter,
    -- PostgreSQL: extract(dow from date) returns 0 for Sunday and 6 for Saturday
    coalesce(a.is_weekend, case when extract(dow from d.date) in (0, 6) then 1 else 0 end) as is_weekend,
    coalesce(a.is_holiday, 0) as is_holiday,
    coalesce(a.is_blackfriday, 0) as is_blackfriday,
    coalesce(a.season_winter, case when extract(month from d.date) in (12, 1, 2) then 1 else 0 end) as season_winter,
    coalesce(a.season_spring, case when extract(month from d.date) in (3, 4, 5) then 1 else 0 end) as season_spring,
    coalesce(a.season_summer, case when extract(month from d.date) in (6, 7, 8) then 1 else 0 end) as season_summer,
    coalesce(a.season_fall,   case when extract(month from d.date) in (9, 10, 11) then 1 else 0 end) as season_fall

from date_spine d
left join attrs a
  on d.date = a.date
