{{ config(materialized='table', file_format='iceberg') }}

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
    select explode(sequence(
        (select start_date from bounds),
        (select end_date from bounds),
        interval 1 day
    )) as date
),

attrs as (
    select *
    from {{ ref('int_date_features') }}
)

select
    d.date,
    year(d.date) as year,
    month(d.date) as month,
    day(d.date) as day,
    dayofweek(d.date) as day_of_week,

    coalesce(a.quarter, quarter(d.date)) as quarter,
    -- In Spark SQL: dayofweek() returns 1 for Sunday, 7 for Saturday
    coalesce(a.is_weekend, case when dayofweek(d.date) in (1, 7) then 1 else 0 end) as is_weekend,
    coalesce(a.is_holiday, 0) as is_holiday,
    coalesce(a.is_blackfriday, 0) as is_blackfriday,
    coalesce(a.season_winter, case when month(d.date) in (12, 1, 2) then 1 else 0 end) as season_winter,
    coalesce(a.season_spring, case when month(d.date) in (3, 4, 5) then 1 else 0 end) as season_spring,
    coalesce(a.season_summer, case when month(d.date) in (6, 7, 8) then 1 else 0 end) as season_summer,
    coalesce(a.season_fall,   case when month(d.date) in (9, 10, 11) then 1 else 0 end) as season_fall

from date_spine d
left join attrs a
  on d.date = a.date
