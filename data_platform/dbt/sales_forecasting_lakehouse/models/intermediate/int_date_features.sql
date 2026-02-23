{{ config(schema='default') }}

with dates as (
  select distinct date from {{ ref('int_active_sales') }}
),

holidays as (
  select date, 1 as is_holiday from {{ ref('stg_holidays') }}
),

blackfriday as (
  select date, 1 as is_blackfriday from {{ ref('stg_blackfriday') }}
)

select
  d.date,
  year(d.date)        as year,
  month(d.date)       as month,
  dayofmonth(d.date)  as day,
  dayofweek(d.date)   as day_of_week,
  quarter(d.date)     as quarter,
  case when dayofweek(d.date) in (1, 7) then 1 else 0 end as is_weekend,
  coalesce(h.is_holiday, 0)       as is_holiday,
  coalesce(bf.is_blackfriday, 0)  as is_blackfriday,
  case when month(d.date) in (12, 1, 2)  then 1 else 0 end as season_winter,
  case when month(d.date) in (3, 4, 5)   then 1 else 0 end as season_spring,
  case when month(d.date) in (6, 7, 8)   then 1 else 0 end as season_summer,
  case when month(d.date) in (9, 10, 11) then 1 else 0 end as season_fall
from dates d
left join holidays    h  on d.date = h.date
left join blackfriday bf on d.date = bf.date
