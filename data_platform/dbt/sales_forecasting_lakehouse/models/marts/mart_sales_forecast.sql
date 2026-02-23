{{ config(file_format='iceberg') }}

select
  s.date,
  s.store_id,
  s.item_id,
  s.units,
  s.log_units,

  -- lag features
  s.logunits_lag_1,
  s.logunits_lag_2,
  s.logunits_lag_3,
  s.logunits_lag_4,
  s.logunits_lag_5,
  s.logunits_lag_6,
  s.logunits_lag_7,
  s.logunits_lag_14,
  s.logunits_lag_21,
  s.logunits_lag_28,

  -- rolling features
  r.roll_avg_7d,  r.roll_min_7d,  r.roll_max_7d,  r.roll_std_7d,
  r.roll_avg_14d, r.roll_min_14d, r.roll_max_14d, r.roll_std_14d,
  r.roll_avg_28d, r.roll_min_28d, r.roll_max_28d, r.roll_std_28d,

  -- ewma (Spark job output — kept external)
  e.ewma7_a05,
  e.ewma7_a075,

  -- store/item context
  a.store_sum_7d,
  a.store_mean_7d,
  a.item_sum_7d,
  a.item_mean_7d,

  -- date features
  d.year, d.month, d.day, d.day_of_week, d.quarter,
  d.is_weekend, d.is_holiday, d.is_blackfriday,
  d.season_winter, d.season_spring, d.season_summer, d.season_fall,

  -- weather features
  w.tmax, w.tmin, w.tavg, w.dewpoint, w.wetbulb,
  w.preciptotal, w.snowfall, w.resultspeed, w.resultdir, w.avgspeed,
  w.is_ra, w.is_sn, w.is_fg, w.is_br, w.is_ts,
  w.is_hz, w.is_dz, w.is_sq, w.is_fz, w.is_vc

from {{ ref('int_sales_with_lags') }} s
left join {{ ref('int_sales_with_rolling') }}    r using (date, store_id, item_id)
left join parquet.`s3a://datalake/intermediate/int_sales_with_ewma` e using (date, store_id, item_id)
left join {{ ref('int_store_item_aggregates') }} a using (date, store_id, item_id)
left join {{ ref('int_date_features') }}         d on s.date = d.date
left join {{ ref('stg_key') }}                   k on s.store_id = k.store_id
left join {{ ref('int_weather_features') }}      w on k.station_id = w.station_id and s.date = w.date
