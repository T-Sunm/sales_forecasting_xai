{{ config(
    materialized='table', 
    indexes=[
      {'columns': ['date', 'store_id', 'item_id']}
    ]
) }}

select
  s.date,
  s.store_id,
  s.item_id,
  s.units,
  s.log_units,

  -- lag features
  s.logunits_lag_1, s.logunits_lag_2, s.logunits_lag_3, s.logunits_lag_4,
  s.logunits_lag_5, s.logunits_lag_6, s.logunits_lag_7, s.logunits_lag_14,
  s.logunits_lag_21, s.logunits_lag_28,

  -- rolling features
  r.roll_avg_7d,  r.roll_min_7d,  r.roll_max_7d,  r.roll_std_7d,
  r.roll_avg_14d, r.roll_min_14d, r.roll_max_14d, r.roll_std_14d,
  r.roll_avg_28d, r.roll_min_28d, r.roll_max_28d, r.roll_std_28d,

  -- ewma
  e.ewma7_a05, e.ewma7_a075,

  -- store/item context
  a.store_sum_7d, a.store_mean_7d, a.item_sum_7d, a.item_mean_7d

from {{ ref('int_sales_with_lags') }} s
left join {{ ref('int_sales_with_rolling') }}    r using (date, store_id, item_id)
left join {{ ref('int_sales_with_ewma') }}       e using (date, store_id, item_id)
left join {{ ref('int_store_item_aggregates') }} a using (date, store_id, item_id)
