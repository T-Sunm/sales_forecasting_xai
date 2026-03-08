{{ config(materialized='table', file_format='iceberg') }}

with weather_base as (
  select
    w.date,
    k.store_id,
    w.station_id,
    -- Numerical Features
    w.tmax, w.tmin, w.tavg, w.depart, w.dewpoint, w.wetbulb, w.heat, w.cool,
    w.sunrise, w.sunset, w.snowfall, w.preciptotal, w.stnpressure, w.sealevel,
    w.resultspeed, w.resultdir, w.avgspeed,
    -- Binary Flags (Weather Codes)
    coalesce(w.is_ra, 0) as is_ra,
    coalesce(w.is_sn, 0) as is_sn,
    coalesce(w.is_fg, 0) as is_fg,
    coalesce(w.is_br, 0) as is_br,
    coalesce(w.is_ts, 0) as is_ts,
    coalesce(w.is_hz, 0) as is_hz,
    coalesce(w.is_dz, 0) as is_dz,
    coalesce(w.is_sq, 0) as is_sq,
    coalesce(w.is_fz, 0) as is_fz,
    coalesce(w.is_vc, 0) as is_vc,
    coalesce(w.is_up, 0) as is_up,
    coalesce(w.is_mi, 0) as is_mi,
    coalesce(w.is_pr, 0) as is_pr,
    coalesce(w.is_bc, 0) as is_bc,
    coalesce(w.is_bl, 0) as is_bl
  from {{ ref('stg_key') }} k
  join {{ ref('int_weather_features') }} w
    on k.station_id = w.station_id
)

select
  b.*,
  p.weather_profile_key
from weather_base b
left join {{ ref('dim_weather_profile') }} p
  on  b.is_ra = p.is_ra
  and b.is_sn = p.is_sn
  and b.is_fg = p.is_fg
  and b.is_br = p.is_br
  and b.is_ts = p.is_ts
  and b.is_hz = p.is_hz
  and b.is_dz = p.is_dz
  and b.is_sq = p.is_sq
  and b.is_fz = p.is_fz
  and b.is_vc = p.is_vc
