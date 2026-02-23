{{ config(schema='default') }}

with base as (
  select date, store_id, item_id, logunits_lag_1
  from {{ ref('int_sales_with_lags') }}
),

store_daily as (
  select store_id, date,
    sum(logunits_lag_1) as store_sum_day,
    avg(logunits_lag_1) as store_mean_day
  from base
  group by store_id, date
),

store_ctx as (
  select store_id, date,
    sum(store_sum_day)  over (partition by store_id order by date rows between 6 preceding and current row) as store_sum_7d,
    avg(store_mean_day) over (partition by store_id order by date rows between 6 preceding and current row) as store_mean_7d
  from store_daily
),

item_daily as (
  select item_id, date,
    sum(logunits_lag_1) as item_sum_day,
    avg(logunits_lag_1) as item_mean_day
  from base
  group by item_id, date
),

item_ctx as (
  select item_id, date,
    sum(item_sum_day)  over (partition by item_id order by date rows between 6 preceding and current row) as item_sum_7d,
    avg(item_mean_day) over (partition by item_id order by date rows between 6 preceding and current row) as item_mean_7d
  from item_daily
)

select
  b.date,
  b.store_id,
  b.item_id,
  sc.store_sum_7d,
  sc.store_mean_7d,
  ic.item_sum_7d,
  ic.item_mean_7d
from base b
left join store_ctx sc on b.store_id = sc.store_id and b.date = sc.date
left join item_ctx  ic on b.item_id  = ic.item_id  and b.date = ic.date
